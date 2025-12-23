# QuantumCredential Demo

> **Status**: Active development for January demo
>
> **Primary Path**: YoshiPi (Linux/ARM64) - leverages "Linux symmetry"
>
> **Future Path**: STM32L5 bare-metal (requires full Farscape/fsnative stack)

---

## The Strategic Pivot

Originally targeting STM32L5 Cortex-M33, the demo pivoted to **YoshiPi** (Raspberry Pi Zero 2 W on carrier board) after recognizing:

| Approach | Risk | Fidelity Stack Required |
|----------|------|------------------------|
| STM32L5 bare-metal | HIGH | Full: Farscape, fsnative, quotation-based memory |
| YoshiPi (Linux) | LOW | Partial: Alex bindings, WebView (same as desktop) |

**The insight**: Both YoshiPi and Desktop are Linux. Same compiler target, same Alloy APIs, same WebView UI. The "embedded demo" becomes a cross-compilation exercise, not a new platform bring-up.

---

## Document Index

| Document | Purpose |
|----------|---------|
| [00_Index.md](./00_Index.md) | Detailed index with component breakdown |
| [01_YoshiPi_Demo_Strategy.md](./01_YoshiPi_Demo_Strategy.md) | Linux symmetry strategy, architecture |
| [02_YoshiPi_Architecture.md](./02_YoshiPi_Architecture.md) | Hardware: Pi Zero 2 W, ADC, GPIO, display |
| [03_Linux_Hardware_Bindings.md](./03_Linux_Hardware_Bindings.md) | Platform.Bindings for GPIO, IIO, USB gadget |
| [04_PostQuantum_Architecture.md](./04_PostQuantum_Architecture.md) | ML-KEM, ML-DSA, entropy conditioning |
| [05_January_Roadmap.md](./05_January_Roadmap.md) | Timeline, sprints, risk assessment |
| [06_Stretch_Goals.md](./06_Stretch_Goals.md) | Sweet Potato Keystation, touch UI, IR transfer |
| [Phase2_STM32L5/](./Phase2_STM32L5/) | Preserved docs for future bare-metal path |

---

## Fidelity Touch Points

### What This Demo Exercises

```
┌─────────────────────────────────────────────────────────────────┐
│  Fidelity Components Exercised by QC Demo                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Alloy     │    │   Alex      │    │   Firefly   │         │
│  │             │    │             │    │   CLI       │         │
│  │ • NativeStr │    │ • Linux     │    │ • ARM64     │         │
│  │ • NativeArr │    │   syscalls  │    │   target    │         │
│  │ • Platform. │    │ • WebView   │    │ • Cross-    │         │
│  │   Bindings  │    │   bindings  │    │   compile   │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            ▼                                    │
│                   Working Native Binary                         │
│                   (Linux/ARM64 ELF)                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### New Dependencies Introduced

| Component | Dependency | Status | Notes |
|-----------|------------|--------|-------|
| **WebView bindings** | WebKitGTK (dynamic) | Design complete | Hand-written, not Farscape-generated |
| **GPIO access** | `/dev/gpiochip0` + ioctl | Design complete | New Platform.Bindings |
| **ADC/IIO access** | sysfs read pattern | Design complete | New Platform.Bindings |
| **USB gadget** | `/dev/ttyGS0` | Existing pattern | Uses writeBytes/readBytes |
| **ARM64 target** | LLVM aarch64 | Needs verification | Target triple change only |
| **PQC algorithms** | ML-KEM, ML-DSA | Research phase | Pure F# or reference impl |

---

## Core Primitives Required

### Must Have for Demo

| Primitive | Location | Current State |
|-----------|----------|---------------|
| `Platform.Bindings.openDevice` | Alloy | **Needs implementation** |
| `Platform.Bindings.ioctl` | Alloy | **Needs implementation** |
| `Platform.Bindings.createWebview` | Alloy | Design complete |
| `Platform.Bindings.setWebviewHtml` | Alloy | Design complete |
| ARM64 syscall emission | Alex | **Needs verification** |
| WebKitGTK MLIR bindings | Alex | **Needs implementation** |

### Workarounds for Demo Day

| Ideal Approach | Demo Workaround | Technical Debt |
|----------------|-----------------|----------------|
| Farscape-generated WebView bindings | Hand-written Platform.Bindings | LOW - clean pattern |
| Full FNCS type resolution | Rely on existing FCS integration | NONE - already works |
| Quotation-based memory model | Stack-only allocation | LOW - valid for demo scope |
| Farscape CMSIS bindings | Linux syscalls instead | NONE - different target |

---

## What's Being "Cheated" vs Built Properly

### Built Properly (No Debt)

- **Platform.Bindings pattern** - Same BCL-free conduit approach
- **Alex binding dispatch** - Data-driven, platform-keyed lookup
- **WebView architecture** - Follows documented desktop stack design
- **Cross-compilation** - Standard LLVM target triple change
- **UI code sharing** - Partas.Solid works unchanged across platforms

### Strategic Shortcuts (Acceptable for Demo)

- **Hand-written bindings** - WebView, GPIO, IIO bindings written manually
  - *Future*: Farscape generates from C headers
  - *Debt*: LOW - patterns are correct, just manual

- **No CMSIS/HAL** - Using Linux syscalls, not bare-metal
  - *Future*: STM32L5 path uses Farscape-generated CMSIS
  - *Debt*: NONE - different platform entirely

- **Reference PQC** - May use existing C implementations initially
  - *Future*: Pure F# implementations
  - *Debt*: MEDIUM - needs native F# for full story

---

## Hardware Insights Captured

### Linux Symmetry
Both demo devices (YoshiPi generator, Sweet Potato keystation) and desktop are Linux. Same toolchain, same APIs, same UI framework.

### Hardware Symmetry
Both embedded devices share identical analog front end:
- Avalanche entropy circuit
- IR transceiver (TX + RX)
- Touchscreen

Role (Generator vs Keystation) is software-defined, not hardware-constrained.

### Self-Sovereign CA
Any device with the analog front end can serve as its own Certificate Authority. This capability is exposed via connected desktop/mobile app, not the touchscreen UI.

---

## Stretch Goal Progression

| Priority | Goal | Effort | Value |
|----------|------|--------|-------|
| 1 | Touch UI on YoshiPi | LOW | WebKitGTK handles it |
| 2 | Sweet Potato Keystation | MEDIUM | Same code, dramatic visual |
| 3 | Entropy visualization | MEDIUM | Canvas + IPC |
| 4 | IR credential transfer | HIGH | Bidirectional, visually striking |
| 5 | QR code transfer | HIGH | Camera + decode complexity |

---

## Success Criteria

### Demo Day Minimum

- [ ] F# compiles to ARM64 Linux ELF
- [ ] Binary runs on YoshiPi
- [ ] ADC samples avalanche circuit
- [ ] PQC credential generated
- [ ] Credential transfers to desktop
- [ ] Desktop verifies signature

### Stretch (If Time Permits)

- [ ] Touch-interactive UI on YoshiPi
- [ ] Sweet Potato as embedded Keystation
- [ ] Real-time entropy visualization
- [ ] IR credential beaming

---

## Related Architecture

| Document | Relevance |
|----------|-----------|
| [Architecture_Canonical.md](../Architecture_Canonical.md) | Platform.Bindings pattern |
| [WebView_Desktop_Architecture.md](../WebView_Desktop_Architecture.md) | WebView stack design |
| [WebView_Build_Integration.md](../WebView_Build_Integration.md) | Firefly build orchestration |
| [Native_Library_Binding_Architecture.md](../Native_Library_Binding_Architecture.md) | Binding design principles |
| [Quotation_Based_Memory_Architecture.md](../Quotation_Based_Memory_Architecture.md) | Future memory model (Phase 2) |

---

## The Demo Narrative

> "Two Linux devices, both running F# compiled to native ARM binaries. The YoshiPi samples quantum noise from an avalanche circuit and generates post-quantum credentials. The Keystation receives and verifies them. Same compiler, same libraries, same UI framework - just different screen sizes.
>
> No runtime. No garbage collector. No managed code vulnerabilities. Just verified, post-quantum security from hardware entropy."

And if asked about the future:
> "The same architecture scales down to bare-metal microcontrollers. That's Phase 2 - same F# code, but compiled through our full Farscape pipeline to run without any OS at all."
