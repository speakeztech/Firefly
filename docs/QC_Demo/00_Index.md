# QuantumCredential Demo Documentation

> **Primary Demo Path**: YoshiPi (Raspberry Pi Zero 2 W on carrier board, running Linux)
>
> **Key Insight**: Both YoshiPi and Desktop are Linux systems. Same Firefly pipeline, same Alloy APIs, same WebView UI.

---

## Phase 1: YoshiPi Demo (PRIMARY)

| # | Document | Purpose |
|---|----------|---------|
| 01 | [YoshiPi_Demo_Strategy](./01_YoshiPi_Demo_Strategy.md) | **START HERE** - Linux symmetry strategy, architecture overview |
| 02 | [YoshiPi_Architecture](./02_YoshiPi_Architecture.md) | Hardware details: Pi Zero 2 W, YoshiPi carrier, quad avalanche circuit |
| 03 | [MLIR_Dialect_Strategy](./03_MLIR_Dialect_Strategy.md) | Compilation path: standard dialects for demo, full vision for future |
| 04 | [Linux_Hardware_Bindings](./04_Linux_Hardware_Bindings.md) | Alloy Platform.Bindings for GPIO, ADC, USB gadget |
| 05 | [PostQuantum_Architecture](./05_PostQuantum_Architecture.md) | PQC algorithms (ML-KEM, ML-DSA), parallel entropy conditioning |
| 06 | [January_Roadmap](./06_January_Roadmap.md) | Timeline, risk assessment, sprint plan |
| 07 | [Stretch_Goals](./07_Stretch_Goals.md) | Sweet Potato Keystation, touch UI, QR/IR transfer |

---

## Demo Components

### YoshiPi: Credential Generator (ARM64 Linux)

- **Hardware**: Raspberry Pi Zero 2 W + YoshiPi carrier board
- **OS**: Raspberry Pi OS (Debian-based)
- **Display**: Integrated touchscreen (480x320)
- **Functions**:
  - Avalanche circuit entropy sampling via ADC
  - PQC key generation (ML-KEM, ML-DSA)
  - Credential signing and framing
  - Transfer via USB gadget serial (or QR/IR as stretch)
- **UI**: WebView (WebKitGTK) with touch - **same code as Keystation**

### Keystation Options

| Platform | Hardware | Display | Status |
|----------|----------|---------|--------|
| **Desktop** | x86_64 Linux | Any monitor | Core demo |
| **Sweet Potato** | Libre Computer (ARM64) | Ultra-wide touchscreen | Stretch goal |

Both run identical code - only the target triple changes.

### Stretch Goal: Embedded Keystation (Sweet Potato)

- **Hardware**: Libre Computer Sweet Potato (2GB RAM, ARM Cortex-A53)
- **Display**: Ultra-wide touchscreen (1280x400 or 320x1480)
- **Functions**: Same as desktop - receive, verify, display, store
- **Visual Impact**: Two embedded devices exchanging quantum-safe credentials

### What They All Share

| Aspect | YoshiPi | Sweet Potato | Desktop |
|--------|---------|--------------|---------|
| OS | Linux/ARM64 | Linux/ARM64 | Linux/x86_64 |
| Alloy APIs | Platform.Bindings | Platform.Bindings | Platform.Bindings |
| WebView | WebKitGTK | WebKitGTK | WebKitGTK |
| UI Code | Partas.Solid | Partas.Solid | Partas.Solid |
| Touch | Yes | Yes | Mouse/optional |

**100% code sharing** across all three platforms.

---

## Phase 2: STM32L5 Path (FUTURE)

The STM32L5 bare-metal path is preserved for post-demo development:

| Document | Purpose |
|----------|---------|
| [Phase2_STM32L5/00_Strategy_Overview](./Phase2_STM32L5/00_Strategy_Overview.md) | NuttX RTOS strategy, POSIX abstraction |
| [Phase2_STM32L5/01_Hardware_Platforms](./Phase2_STM32L5/01_Hardware_Platforms.md) | STM32L5 NUCLEO, CMSIS details |
| [Phase2_STM32L5/02_Farscape_Assessment](./Phase2_STM32L5/02_Farscape_Assessment.md) | Gap analysis for CMSIS binding generation |
| [Phase2_STM32L5/03_UI_Options](./Phase2_STM32L5/03_UI_Options.md) | LVGL, embedded UI alternatives |

Phase 2 requires:
- Full quotation-based memory architecture (Farscape)
- CMSIS HAL bindings via Farscape codegen
- NuttX RTOS integration or bare-metal execution

---

## Related Architecture Documents

These documents in the parent `/docs/` folder provide architectural context:

### Desktop UI Stack
- [WebView_Desktop_Architecture.md](../WebView_Desktop_Architecture.md) - WebView stack architecture
- [WebView_Build_Integration.md](../WebView_Build_Integration.md) - Firefly build orchestration
- [WebView_Desktop_Design.md](../WebView_Desktop_Design.md) - Callbacks, IPC design

### Core Architecture
- [Architecture_Canonical.md](../Architecture_Canonical.md) - Core Firefly architecture, platform bindings
- [Quotation_Based_Memory_Architecture.md](../Quotation_Based_Memory_Architecture.md) - Memory model (Phase 2)
- [Native_Library_Binding_Architecture.md](../Native_Library_Binding_Architecture.md) - Binding design principles
