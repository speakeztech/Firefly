# QuantumCredential Demo

The QuantumCredential demo brings together hardware entropy, post-quantum cryptography, and native F# compilation in a way that makes the abstract tangible. A small device samples quantum noise from an avalanche circuit, conditions that entropy, and generates cryptographic credentials that will remain secure even when quantum computers reach maturity. A second device receives those credentials, verifies the signatures, and stores them for later use. Both devices run the same F# code, compiled to native ARM binaries, displaying their status through WebView-based touchscreen interfaces.

The architectural foundation of this demo rests on symmetry. The credential generator and the keystation share identical hardware: the same avalanche circuit, the same IR transceiver, the same touchscreen. The only difference between them is software configuration. Because both devices run Linux (Raspberry Pi OS on the YoshiPi carrier board), they share the same compilation target as the desktop development machine. The Firefly compiler that produces x86_64 Linux binaries produces ARM64 Linux binaries with nothing more than a target triple change. The Alloy library that provides console I/O and timing on desktop provides GPIO and ADC access on the embedded device through the same Platform.Bindings pattern. The WebKitGTK library that renders the desktop UI renders the touchscreen UI with identical Partas.Solid components.

This recognition that both platforms are simply Linux transformed what could have been a high-risk embedded systems project into a tractable cross-compilation exercise. The STM32L5 bare-metal path remains documented for future work, but the demo will ship on hardware that allows the team to focus on the cryptographic and UI narrative rather than contending with RTOS integration.

---

## Document Overview

### [01_YoshiPi_Demo_Strategy.md](./01_YoshiPi_Demo_Strategy.md)

This document establishes the strategic foundation for the demo. It explains why running Debian on a Raspberry Pi Zero 2 W represents an advantage rather than a compromise, walking through the symmetric architecture where the same Firefly pipeline, the same Alloy APIs, and the same WebView approach work identically on the embedded device and the desktop. Comparison tables showing what components are shared across platforms make the "write once, compile twice" proposition concrete.

### [02_YoshiPi_Architecture.md](./02_YoshiPi_Architecture.md)

This document tells the hardware story. The YoshiPi carrier board provides analog inputs for the avalanche circuit, GPIO for status LEDs, and a touchscreen for the monitor UI. The document details the physical connections, the Linux device interfaces (`/dev/gpiochip0` for GPIO, `/sys/bus/iio/devices/` for ADC), and how the display subsystem options map to the UI architecture. WebKitGTK is the preferred rendering path, with framebuffer access available as a fallback. ASCII diagrams showing the avalanche circuit connection and the process memory layout ground the abstract in the physical.

### [03_Linux_Hardware_Bindings.md](./03_Linux_Hardware_Bindings.md)

This document serves as the bridge between Alloy and Linux hardware. Platform.Bindings functions such as `openDevice`, `ioctl`, and `readBytes` provide the vocabulary for hardware access. The document shows how GPIO control flows from F# through the binding pattern to Linux ioctl calls, how ADC sampling reads from sysfs files, and how USB gadget mode enables credential transfer. Code examples are complete enough to implement directly, with the understanding that Alex provides the platform-specific syscall emission.

### [04_PostQuantum_Architecture.md](./04_PostQuantum_Architecture.md)

This document addresses the cryptographic core of the system. ML-KEM provides key encapsulation, ML-DSA provides digital signatures, and SHAKE-256 provides entropy conditioning. The document covers the NIST-selected algorithms, their security levels, and how hardware entropy from the avalanche circuit seeds the entire cryptographic pipeline. The credential structure and signing flow demonstrate how these pieces compose into a complete post-quantum credential system.

### [05_January_Roadmap.md](./05_January_Roadmap.md)

This document contains the timeline and risk assessment. It includes sprint breakdowns, dependency identification, and contingency planning for demo day. The document maintains honesty about what is known, what remains uncertain, and what would constitute acceptable fallback positions if stretch goals do not materialize.

### [06_Stretch_Goals.md](./06_Stretch_Goals.md)

This document describes the vision beyond the minimum viable demo. Topics include the Libre Sweet Potato as an embedded keystation with an ultra-wide touchscreen, touch interaction that WebKitGTK provides without additional implementation effort, IR credential transmission between devices, and real-time entropy visualization. The document also captures deeper architectural observations: that both devices share identical analog front ends (making their roles purely software-defined), and that any device with the entropy circuit can serve as its own certificate authority. The CA capability remains accessible through the connected desktop interface rather than the touchscreen UI, but its presence transforms the narrative from "credential generator" to "decentralized PKI infrastructure."

### [Phase2_STM32L5/](./Phase2_STM32L5/)

This directory preserves the future path. When the demo ships and attention turns to bare-metal targets, these documents provide the starting point: NuttX RTOS integration, Farscape-generated CMSIS bindings, and the full quotation-based memory architecture. This preparatory work informed the YoshiPi approach and will benefit from lessons learned during demo development.

---

## Architectural Integration

The reactive UI model will be familiar to developers who have worked with Elmish, MVU, or similar architectures. Partas.Solid compiles F# component definitions to SolidJS, which executes in WebKitGTK's JavaScript engine. State management follows the signal and effect pattern: `createSignal` for reactive state, `createEffect` for side effects, and `createStore` for complex state trees. Developers accustomed to TanStack-style store patterns will find the mental model of reactive derivations and fine-grained updates translates directly.

The native backend communicates with the UI through WebView bindings. When a button tap needs to trigger credential generation, the JavaScript handler calls a bound function that the native code registered at startup. The native code performs the computationally intensive work: sampling the ADC, executing the PQC algorithms, and framing the credential for transfer. Results return to the UI for display. This separation keeps the UI responsive while cryptographic operations execute at native speed.

The Platform.Bindings pattern provides hardware abstraction. Functions in Alloy's `Platform.Bindings` module carry placeholder implementations using `Unchecked.defaultof<T>`. Alex recognizes these declarations during compilation and emits platform-specific code: Linux syscalls for the YoshiPi, different syscalls or API calls for other platforms. This pattern extends naturally to new hardware interfaces. Adding support for a new peripheral requires declaring the binding signature in Alloy and implementing the corresponding emission logic in Alex.

---

## Fidelity Components Exercised

| Component | Capability | Application in Demo |
|-----------|------------|---------------------|
| **Firefly CLI** | ARM64 cross-compilation | Target triple: `aarch64-unknown-linux-gnu` |
| **Alloy** | Native types and Platform.Bindings | NativeStr, NativeArray, GPIO and ADC bindings |
| **Alex** | Linux syscall emission and library bindings | ioctl, read, write, and WebKitGTK function calls |
| **Partas.Solid** | Reactive UI components | Touchscreen monitor interface |
| **BAREWire** | Binary serialization | Credential framing for transfer |

### New Platform.Bindings Required

```
Platform.Bindings.openDevice    : nativeint -> int -> int
Platform.Bindings.closeDevice   : int -> int
Platform.Bindings.ioctl         : int -> uint64 -> nativeint -> int
Platform.Bindings.createWebview : int -> nativeint -> nativeint
Platform.Bindings.setWebviewHtml: nativeint -> nativeint -> int
```

These bindings follow the established pattern. The binding declarations reside in Alloy; Alex provides platform-specific emission for each target.

---

## Technical Debt Assessment

### Components Built According to Architecture

The Platform.Bindings pattern, the Alex dispatch architecture, the WebView integration design, and the cross-compilation approach all follow the documented Fidelity architecture. Work completed for this demo directly informs and validates the design for future platforms.

### Expedient Choices for Demo Timeline

WebView and hardware bindings are hand-written rather than Farscape-generated. PQC algorithms may initially wrap reference C implementations rather than pure F# implementations. These shortcuts follow correct patterns (binding signatures are correct, dispatch logic is correct) but involve manual implementation work that would eventually be automated through Farscape code generation.

### Work Deferred to Phase 2

The STM32L5 bare-metal path requires the full quotation-based memory architecture: Farscape generating bindings from CMSIS headers, fsnative nanopasses validating memory constraints, and complete stack-discipline enforcement. That infrastructure is not required for the Linux-based demo and remains future work.

---

## The Demo Narrative

Two small devices sit on a table. One samples quantum noise and generates credentials. The other receives and verifies them. Both display their status on touchscreens. An IR beam carries credentials from one device to the other.

Beneath the surface: F# compiled to native ARM binaries with no runtime and no garbage collector. Hardware entropy derived from avalanche breakdown in reverse-biased semiconductor junctions. Post-quantum cryptography designed to survive the transition to quantum computing. A UI framework that executes identically on embedded touchscreens and desktop monitors.

The demo answers a question that initially sounds impossible: can a statically-typed functional language with rich type inference compile to efficient native code while providing memory safety guarantees? The answer is yes. Here is the device running it.

---

## Document Navigation

| Purpose | Document |
|---------|----------|
| Understand the overall strategy | [01_YoshiPi_Demo_Strategy.md](./01_YoshiPi_Demo_Strategy.md) |
| Review hardware specifications | [02_YoshiPi_Architecture.md](./02_YoshiPi_Architecture.md) |
| Implement hardware bindings | [03_Linux_Hardware_Bindings.md](./03_Linux_Hardware_Bindings.md) |
| Study the cryptographic design | [04_PostQuantum_Architecture.md](./04_PostQuantum_Architecture.md) |
| Review timeline and milestones | [05_January_Roadmap.md](./05_January_Roadmap.md) |
| Explore stretch goals | [06_Stretch_Goals.md](./06_Stretch_Goals.md) |
| Prepare for bare-metal targets | [Phase2_STM32L5/](./Phase2_STM32L5/) |
