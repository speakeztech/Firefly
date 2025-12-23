# QuantumCredential Demo

The QuantumCredential demo brings together hardware entropy, post-quantum cryptography, and native F# compilation in a way that makes the abstract tangible. A small device samples quantum noise from an avalanche circuit, conditions that entropy, and generates cryptographic credentials that will remain secure even when quantum computers mature. Another device receives those credentials, verifies the signatures, and stores them. Both devices run the same F# code, compiled to native ARM binaries, displaying their status through WebView-based touchscreen interfaces.

What makes this demo architecturally interesting is the symmetry. The credential generator and the keystation share identical hardware - the same avalanche circuit, the same IR transceiver, the same touchscreen. The only difference is software configuration. And because both devices run Linux (Raspberry Pi OS on the YoshiPi carrier board), they share the same compilation target as the desktop development machine. The Firefly compiler that produces x86_64 Linux binaries produces ARM64 Linux binaries with a target triple change. The Alloy library that provides console I/O and timing on desktop provides GPIO and ADC access on the embedded device through the same Platform.Bindings pattern. The WebKitGTK library that renders the desktop UI renders the touchscreen UI with identical Partas.Solid components.

This "it's just Linux" insight transformed what could have been a high-risk embedded systems project into a tractable cross-compilation exercise. The STM32L5 bare-metal path remains documented for future work, but the demo will ship on hardware that lets us focus on the cryptographic and UI story rather than fighting with RTOS bring-up.

---

## What's in This Folder

### [01_YoshiPi_Demo_Strategy.md](./01_YoshiPi_Demo_Strategy.md)
The strategic foundation. This document explains why running Debian on a Raspberry Pi Zero 2 W is not a compromise but an advantage. It walks through the symmetric architecture where the same Firefly pipeline, the same Alloy APIs, and the same WebView approach work identically on the embedded device and the desktop. The comparison tables showing what's shared across platforms make the "write once, compile twice" story concrete.

### [02_YoshiPi_Architecture.md](./02_YoshiPi_Architecture.md)
The hardware story. The YoshiPi carrier board provides analog inputs for the avalanche circuit, GPIO for status LEDs, and a touchscreen for the monitor UI. This document details the physical connections, the Linux device interfaces (`/dev/gpiochip0`, `/sys/bus/iio/devices/`), and how the display subsystem options (WebKitGTK preferred, framebuffer fallback) map to the UI architecture. The ASCII diagrams showing the avalanche circuit connection and the memory layout ground the abstract in the physical.

### [03_Linux_Hardware_Bindings.md](./03_Linux_Hardware_Bindings.md)
The bridge between Alloy and Linux hardware. Platform.Bindings functions like `openDevice`, `ioctl`, and `readBytes` provide the vocabulary for hardware access. This document shows how GPIO control flows from F# through the binding pattern to Linux ioctl calls, how ADC sampling reads from sysfs files, and how USB gadget mode enables credential transfer. The code examples are complete enough to implement, with the understanding that Alex provides the syscall emission.

### [04_PostQuantum_Architecture.md](./04_PostQuantum_Architecture.md)
The cryptographic heart. ML-KEM for key encapsulation, ML-DSA for digital signatures, SHAKE-256 for entropy conditioning. This document covers the NIST-selected algorithms, their security levels, and how hardware entropy from the avalanche circuit seeds the entire cryptographic pipeline. The credential structure and signing flow show how the pieces compose into a complete post-quantum credential system.

### [05_January_Roadmap.md](./05_January_Roadmap.md)
The timeline and risk assessment. Sprint breakdowns, dependency identification, and contingency planning for demo day. This document is honest about what's known, what's uncertain, and what would constitute acceptable fallback positions if stretch goals don't materialize.

### [06_Stretch_Goals.md](./06_Stretch_Goals.md)
The vision beyond minimum viable demo. The Libre Sweet Potato as an embedded keystation with an ultra-wide touchscreen. Touch interaction that WebKitGTK provides for free. IR credential beaming between devices. Real-time entropy visualization. This document also captures the deeper insights: that both devices share identical analog front ends (making roles software-defined), and that any device with the entropy circuit can serve as its own certificate authority. The CA capability stays hidden behind the connected desktop interface, but it transforms the narrative from "credential generator" to "decentralized PKI infrastructure."

### [Phase2_STM32L5/](./Phase2_STM32L5/)
The future path, preserved. When the demo ships and attention turns to bare-metal targets, these documents provide the starting point. NuttX RTOS integration, Farscape-generated CMSIS bindings, the full quotation-based memory architecture. This work informed the YoshiPi approach and will benefit from lessons learned during demo development.

---

## How the Pieces Connect

The reactive UI model will feel familiar to anyone who has worked with Elmish, MVU, or similar architectures. Partas.Solid compiles F# component definitions to SolidJS, which runs in WebKitGTK's JavaScript engine. State management follows the signal/effect pattern - `createSignal` for reactive state, `createEffect` for side effects, `createStore` for complex state trees. The TanStack-style store patterns work naturally here; the mental model of reactive derivations and fine-grained updates translates directly.

The native backend communicates with the UI through WebView bindings. When a button tap needs to trigger credential generation, the JavaScript handler calls a bound function that the native code registered. The native code does the heavy lifting - sampling the ADC, running the PQC algorithms, framing the credential - and sends results back to the UI for display. This split keeps the UI responsive while cryptographic operations run at native speed.

The Platform.Bindings pattern provides the hardware abstraction. Functions in Alloy's `Platform.Bindings` module have placeholder implementations (`Unchecked.defaultof<T>`). Alex recognizes these during compilation and emits platform-specific code - Linux syscalls for the YoshiPi, different syscalls or API calls for other platforms. This pattern extends naturally to new hardware interfaces: declare the binding signature in Alloy, implement the emission in Alex.

---

## What This Demo Exercises in Fidelity

| Component | Capability | Demo Usage |
|-----------|------------|------------|
| **Firefly CLI** | ARM64 cross-compilation | Target triple: `aarch64-unknown-linux-gnu` |
| **Alloy** | Native types, Platform.Bindings | NativeStr, NativeArray, GPIO/ADC bindings |
| **Alex** | Linux syscall emission, WebView bindings | ioctl, read, write + webview library calls |
| **Partas.Solid** | Reactive UI components | Touchscreen monitor interface |
| **BAREWire** | Binary serialization | Credential framing for transfer |

### New Bindings Required

```
Platform.Bindings.openDevice    : nativeint -> int -> int
Platform.Bindings.closeDevice   : int -> int
Platform.Bindings.ioctl         : int -> uint64 -> nativeint -> int
Platform.Bindings.createWebview : int -> nativeint -> nativeint
Platform.Bindings.setWebviewHtml: nativeint -> nativeint -> int
```

These follow the established pattern. The binding declarations exist in Alloy; Alex provides platform-specific emission.

---

## Honest Accounting

### What's Built Properly

The Platform.Bindings pattern, the Alex dispatch architecture, the WebView integration design, the cross-compilation approach - these follow the documented Fidelity architecture. The work done here directly informs and validates the design for future platforms.

### What's Expedient for Demo Day

WebView and hardware bindings are hand-written rather than Farscape-generated. PQC algorithms may initially wrap reference C implementations rather than pure F#. These shortcuts follow correct patterns (the binding signatures are right, the dispatch logic is right) but involve manual implementation that would eventually be automated.

### What's Deferred to Phase 2

The STM32L5 bare-metal path requires the full quotation-based memory architecture: Farscape generating bindings from CMSIS headers, fsnative nanopasses validating memory constraints, the complete stack-discipline enforcement. That infrastructure isn't needed for the Linux-based demo and remains future work.

---

## The Narrative

Two small devices sit on a table. One samples quantum noise and generates credentials. The other receives and verifies them. Both show their status on touchscreens. An IR beam carries the credential from one to the other.

Under the surface: F# compiled to native ARM binaries. No runtime, no garbage collector. Hardware entropy from avalanche breakdown. Post-quantum cryptography that will survive the quantum computing transition. A UI framework that runs identically on embedded touchscreens and desktop monitors.

The demo answers a question that sounds impossible: can a statically-typed functional language with rich type inference compile to efficient bare-metal code while providing memory safety guarantees? The answer is yes, and here's the device running it.

---

## Quick Links

| Need | Document |
|------|----------|
| Understand the strategy | [01_YoshiPi_Demo_Strategy.md](./01_YoshiPi_Demo_Strategy.md) |
| See the hardware details | [02_YoshiPi_Architecture.md](./02_YoshiPi_Architecture.md) |
| Implement hardware bindings | [03_Linux_Hardware_Bindings.md](./03_Linux_Hardware_Bindings.md) |
| Understand the cryptography | [04_PostQuantum_Architecture.md](./04_PostQuantum_Architecture.md) |
| Check the timeline | [05_January_Roadmap.md](./05_January_Roadmap.md) |
| Explore stretch goals | [06_Stretch_Goals.md](./06_Stretch_Goals.md) |
| Plan for bare-metal future | [Phase2_STM32L5/](./Phase2_STM32L5/) |
