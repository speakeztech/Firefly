# QuantumCredential Demo

## Business Context

Enterprise authentication faces a fundamental crisis. Passwords remain the weakest link in security infrastructure, while network-dependent two-factor authentication has proven vulnerable to SIM-swapping, SS7 exploits, and carrier-level compromises. The quantum computing threat compounds these concerns: cryptographic keys generated and distributed through conventional means today may be harvested for decryption tomorrow when quantum computers mature. Organizations handling sensitive data require authentication solutions that operate independently of vulnerable network infrastructure while providing protection against both current and future cryptographic threats.

The QuantumCredential device addresses these requirements through hardware-anchored security. It generates cryptographic credentials using entropy derived from quantum physical processes in an avalanche circuit, implements NIST-approved post-quantum algorithms (ML-KEM, ML-DSA), and operates completely air-gapped from networks during credential generation and distribution. The KeyStation provides the complementary verification and storage interface, receiving credentials through out-of-band channels (QR code, infrared) that never traverse vulnerable network paths.

Two patents pending protect the core innovations in this architecture:

- **US 63/780,027**: "Air-Gapped Dual Network Architecture for QRNG Cryptographic Certificate Distribution via QR Code and Infrared Transfer in WireGuard Overlay Networks"
- **US 63/780,055**: "Quantum-Resistant Hardware Security Module with Decentralized Identity Capabilities"

This demo validates the end-to-end functionality of these patented innovations using the Fidelity framework.

## Technology Foundations

The QuantumCredential demo serves as a proving ground for the Fidelity framework's core architectural principles. While the demo itself runs on Linux-based hardware for pragmatic reasons, the underlying technology foundations apply equally to bare-metal microcontrollers, mobile devices, and datacenter systems.

### Quotation-Based Memory Architecture

At the heart of the Fidelity framework lies a novel approach to memory management where F# quotations and active patterns serve as the foundational substrate for type-safe hardware access. This is not a peripheral feature or stretch goal; it represents the central architectural innovation that distinguishes Fidelity from other compilation approaches.

Quotations provide inspectable, transformable representations of memory constraints:

```fsharp
let GPIO_ODR_Constraint = <@
    { Address = 0x48000014un
      Region = Peripheral
      Access = ReadWrite
      Volatile = true
      Width = Bits32 }
@>
```

Active patterns provide computed pattern matching that recognizes hardware operations:

```fsharp
let (|VolatilePeripheralWrite|_|) (node: PSGNode) =
    match node with
    | FunctionCall (WriteOp, [ptr; value]) ->
        match ptr.MemoryConstraint with
        | Some <@ { Volatile = true; Region = Peripheral } @> ->
            Some (VolatilePeripheralWrite (ptr, value))
        | _ -> None
    | _ -> None
```

This architecture enables compile-time validation of memory access patterns, ensuring that read-only registers cannot be written, volatile accesses generate appropriate memory barriers, and hardware constraints are enforced through the type system rather than runtime checks.

The demo exercises this architecture through GPIO control, ADC sampling, and WebView rendering, validating the quotation flow from Alloy declarations through fsnative nanopasses to Alex emission.

### BAREWire Zero-Copy Protocol

The BAREWire protocol, protected by patent US 63/786,247 ("System and Method for Zero-Copy Inter-Process Communication Using BARE Protocol"), provides the serialization and memory management infrastructure for credential transfer. BAREWire enables type-safe binary communication without allocation overhead, critical for both embedded credential generation and high-performance datacenter applications.

### Platform Binding Pattern

The demo validates the Platform.Bindings pattern where Alloy declares hardware access signatures without implementation, and Alex provides platform-specific MLIR emission. This separation ensures that application code remains hardware-agnostic while enabling aggressive platform-specific optimization:

```fsharp
// Alloy declares the interface
module Platform.Bindings =
    let ioctl (fd: int) (request: uint64) (arg: nativeint) : int =
        Unchecked.defaultof<int>

// Alex emits platform-specific syscalls
// Linux ARM64: svc #0 with appropriate register setup
// Linux x86_64: syscall instruction
// Future bare-metal: direct register manipulation
```

## Hardware Strategy

The demo runs on the YoshiPi carrier board (Raspberry Pi Zero 2 W running Debian) rather than bare-metal microcontrollers. This choice reflects pragmatic risk management for demo timelines while preserving the architectural validity of the demonstration.

Both the YoshiPi (credential generator) and the Sweet Potato (keystation) are Linux/ARM64 systems. They share the same compilation target as desktop development machines, enabling code sharing across the UI, cryptographic, and communication layers. The Firefly compiler produces ARM64 binaries with only a target triple change from x86_64 development.

The hardware platforms share identical analog front ends:
- Avalanche circuit for quantum-grade entropy generation
- Infrared transceiver for air-gapped credential transfer
- Touchscreen interface for user interaction
- ADC inputs for entropy sampling

This symmetry means the role distinction between credential generator and keystation is purely software-defined. Any device with the entropy circuit can serve as its own certificate authority, a capability exposed through the connected desktop interface for advanced users.

The STM32L5 bare-metal path remains documented in the Phase2_STM32L5 subdirectory. That approach requires the full quotation-based memory architecture with Farscape-generated CMSIS bindings and represents the post-demo development trajectory.

## Document Index

### Core Documentation

**[01_YoshiPi_Demo_Strategy.md](./01_YoshiPi_Demo_Strategy.md)** establishes the symmetric architecture where credential generator and keystation share code through common Linux targeting. The document explains how Alloy APIs, WebView rendering, and Platform.Bindings work identically across both devices.

**[02_YoshiPi_Architecture.md](./02_YoshiPi_Architecture.md)** details the hardware integration including the avalanche circuit connection to ADC inputs, GPIO control via the Linux gpiochip interface, and display rendering through WebKitGTK. Memory layout diagrams show how stack-based allocation serves the demo's needs.

**[03_Linux_Hardware_Bindings.md](./03_Linux_Hardware_Bindings.md)** documents the Platform.Bindings extensions for Linux hardware access: device file operations, ioctl for GPIO control, sysfs reading for ADC sampling, and USB gadget mode for credential transfer. Code examples demonstrate the quotation-based constraint pattern even within the Linux context.

**[04_PostQuantum_Architecture.md](./04_PostQuantum_Architecture.md)** covers the cryptographic design: ML-KEM for key encapsulation, ML-DSA for digital signatures, SHAKE-256 for entropy conditioning. The credential structure and signing flow implement the patent-protected air-gapped distribution architecture.

**[05_January_Roadmap.md](./05_January_Roadmap.md)** contains timeline planning, sprint structure, risk assessment, and contingency strategies for demo delivery.

**[06_Stretch_Goals.md](./06_Stretch_Goals.md)** describes enhanced demo capabilities including real-time entropy visualization, bidirectional IR credential transfer, and the self-sovereign CA functionality that transforms individual devices into decentralized PKI infrastructure.

### Future Development

**[Phase2_STM32L5/](./Phase2_STM32L5/)** preserves documentation for the bare-metal path: NuttX RTOS integration, Farscape-generated CMSIS bindings, and the complete quotation-based memory architecture operating without an OS layer.

## Fidelity Components Exercised

| Component | Capability Validated |
|-----------|---------------------|
| **Firefly** | ARM64 cross-compilation, build orchestration |
| **Alloy** | NativeStr, NativeArray, Platform.Bindings pattern |
| **Alex** | Linux syscall emission, WebKitGTK library bindings |
| **BAREWire** | Credential serialization, memory-mapped descriptors |
| **fsnative** | Quotation attachment, constraint validation nanopasses |

The demo provides concrete validation that the architectural principles documented in Quotation_Based_Memory_Architecture.md operate correctly in practice, even when the target platform is Linux rather than bare-metal. The nanopass pipeline processes quotation-based constraints identically regardless of whether the ultimate emission is a Linux syscall or a direct register write.

## Success Criteria

The demo succeeds when:
- F# code compiles to functional ARM64 Linux binaries
- Avalanche circuit sampling produces quality entropy
- Post-quantum credential generation completes correctly
- Credentials transfer via USB (and optionally IR) to the keystation
- Signatures verify correctly on the receiving device
- The WebView UI displays status on touchscreen interfaces

These criteria validate both the immediate demo goals and the underlying architectural foundations that will extend to bare-metal targets, heterogeneous computing environments, and the broader Fidelity framework vision.

## Related Architecture Documents

| Document | Relevance |
|----------|-----------|
| [Quotation_Based_Memory_Architecture.md](../Quotation_Based_Memory_Architecture.md) | Core memory model underlying all Fidelity targets |
| [Architecture_Canonical.md](../Architecture_Canonical.md) | Platform.Bindings pattern and nanopass pipeline |
| [WebView_Desktop_Architecture.md](../WebView_Desktop_Architecture.md) | WebView integration shared between demo devices |
| [Native_Library_Binding_Architecture.md](../Native_Library_Binding_Architecture.md) | Binding design principles for hardware access |

## Intellectual Property

This demo implements technology protected by the following pending patents:

| Patent Application | Title | Relevance |
|-------------------|-------|-----------|
| US 63/780,027 | Air-Gapped Dual Network Architecture for QRNG Cryptographic Certificate Distribution | Core credential distribution architecture |
| US 63/780,055 | Quantum-Resistant Hardware Security Module with Decentralized Identity Capabilities | HSM and DID functionality |
| US 63/786,247 | System and Method for Zero-Copy Inter-Process Communication Using BARE Protocol | BAREWire serialization |
| US 63/786,264 | System and Method for Verification-Preserving Compilation Using Formal Certificate Guided Optimization | Fidelity compilation approach |

The demo validates these innovations in a working implementation suitable for investor presentation and technical due diligence.
