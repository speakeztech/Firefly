# Parallel XOR Entropy Combination: Design and Mathematical Foundation

## Executive Summary

This document describes a method for generating high-quality random bytes from four independent avalanche noise circuits using parallel XOR combination. The approach eliminates the need for Von Neumann debiasing or whitening algorithms by leveraging the mathematical properties of XOR across independent entropy sources. Each ADC sample cycle produces one byte of output directly, maximizing throughput even on I2C or SPI-constrained hardware.

---

## Part 1: The Problem with Single-Source Entropy

### Why Raw Samples Aren't Enough

A single avalanche noise source, while quantum in nature, may exhibit bias. Physical factors contribute to imperfect randomness:

- **DC offset variations** cause the signal to spend more time above or below center
- **Thermal drift** shifts the operating point over time
- **Component tolerances** create systematic bias in the noise distribution
- **ADC nonlinearity** introduces quantization bias at certain code points

A source that produces 55% ones and 45% zeros has a 5% bias. For cryptographic applications, this is unacceptable—an attacker could exploit any statistical deviation from perfect randomness.

### Traditional Solutions and Their Costs

**Von Neumann Debiasing**: Read pairs of bits; output only when they differ (01→0, 10→1, discard 00 and 11). This guarantees unbiased output but discards roughly 75% of samples when bias exists, and 50% even with perfect sources. It fundamentally cannot exceed 50% efficiency.

**Whitening Functions**: Apply cryptographic transforms (SHA-256, AES) to spread entropy. This adds computational overhead and requires trust in the whitening algorithm itself—problematic for a hardware random number generator whose purpose is to avoid algorithmic dependencies.

**Our Approach**: Combine multiple independent sources with XOR. No samples discarded. No algorithmic transforms. Pure bitwise combination with mathematical guarantees.

---

## Part 2: XOR as an Entropy Combiner

### The Fundamental Property

XOR (exclusive or) has a remarkable characteristic: **if either input is random, the output is random**.

The operation is defined as:
```
0 ⊕ 0 = 0
0 ⊕ 1 = 1
1 ⊕ 0 = 1
1 ⊕ 1 = 0
```

Consider two bit sources A and B where A is perfectly random (50/50) and B is heavily biased (90% ones). The XOR output remains perfectly random:

- When A=0 (50% of the time): output equals B
- When A=1 (50% of the time): output equals NOT B

Since A acts as a random selector between B and its complement, the output is 50/50 regardless of B's bias. The randomness of A "masks" any bias in B.

This property means XOR **never degrades entropy**—it can only preserve or improve it.

### Quantifying Bias Reduction

Let ε (epsilon) represent the deviation from perfect balance. A source with bias ε produces:
- P(one) = 0.5 + ε
- P(zero) = 0.5 - ε

For example, ε = 0.05 means 55% ones, 45% zeros—a 5% bias.

When two independent sources with bias ε are XOR'd, we can calculate the output bias:

```
P(A⊕B = 1) = P(A=1)×P(B=0) + P(A=0)×P(B=1)
           = (0.5 + ε)(0.5 - ε) + (0.5 - ε)(0.5 + ε)
           = 2 × (0.5 + ε)(0.5 - ε)
           = 2 × (0.25 - ε²)
           = 0.5 - 2ε²
```

The output bias is **2ε²**, which is dramatically smaller than ε for any reasonable bias level.

---

## Part 3: The Four-Channel Parallel Tree

### Tree Structure vs Serial Chain

With four channels, we could XOR serially:
```
Serial: CH0 → ⊕ → ⊕ → ⊕ → result
             ↑    ↑    ↑
            CH1  CH2  CH3

Depth: 3 operations in sequence
```

Or we could XOR in a parallel tree:
```
Parallel:
    CH0 ──┐
          ⊕ ──┐
    CH1 ──┘   │
              ⊕ ──→ result
    CH2 ──┐   │
          ⊕ ──┘
    CH3 ──┘

Depth: 2 levels (log₂ of 4)
```

The parallel tree has two critical advantages:

1. **Latency**: Only 2 XOR operations on the critical path instead of 3
2. **Parallelism**: The two Level-1 XORs are independent and can execute simultaneously

### Bias Reduction in the Tree

**Level 1** (two parallel operations):
- CH0 ⊕ CH1 produces output with bias 2ε²
- CH2 ⊕ CH3 produces output with bias 2ε²

**Level 2** (combining the results):
- Two sources with bias 2ε² are XOR'd
- Output bias = 2 × (2ε²)² = 2 × 4ε⁴ = **8ε⁴**

### Numerical Impact

Starting with ε = 0.05 (5% bias per channel):

| Configuration | Bias Formula | Bias Value | As Percentage |
|---------------|--------------|------------|---------------|
| Single channel | ε | 0.05 | 5% |
| Two channels | 2ε² | 0.005 | 0.5% |
| Four channels (tree) | 8ε⁴ | 0.00005 | 0.005% |

The four-channel tree reduces 5% bias to 0.005%—a factor of 1000 improvement. This approaches cryptographic quality without any algorithmic intervention.

### Why Independence Matters

The mathematical analysis assumes the four sources are **independent**—the noise in one channel has no correlation with noise in another. This requires:

- **Separate avalanche diodes**: Each channel has its own Zener, not shared
- **Isolated bias networks**: No common current paths that could create correlation
- **Independent power filtering**: Decoupling capacitors per channel

If channels were correlated, the XOR would not achieve the full ε⁴ reduction. For example, if CH0 and CH1 produced identical values, their XOR would always be zero—no entropy at all.

---

## Part 4: Byte-Level Operation

### From Bits to Bytes

The ADC produces 10-bit samples. We extract the lower 8 bits from each channel and XOR them:

```
CH0 sample: 0x15A → 0x5A (lower 8 bits)
CH1 sample: 0x23C → 0x3C
CH2 sample: 0x0A7 → 0xA7
CH3 sample: 0x1E1 → 0xE1
```

The parallel XOR tree operates on all 8 bits simultaneously:

```
Level 1:
  0x5A ⊕ 0x3C:
    01011010
  ⊕ 00111100
  ──────────
    01100110 = 0x66

  0xA7 ⊕ 0xE1:
    10100111
  ⊕ 11100001
  ──────────
    01000110 = 0x46

Level 2:
  0x66 ⊕ 0x46:
    01100110
  ⊕ 01000110
  ──────────
    00100000 = 0x20
```

Each bit position in the output byte combines four independent entropy sources. The ε⁴ bias reduction applies to every bit.

### One Byte Per Sample Cycle

Unlike Von Neumann debiasing (which discards samples) or bit-at-a-time approaches, this method produces exactly one output byte for every four-channel read cycle:

- Read CH0, CH1, CH2, CH3 (one ADC cycle each)
- XOR in parallel tree (2 levels of bitwise XOR)
- Output: 1 byte

For 4096 bytes of entropy, we need exactly 4096 sample cycles. No waste, no retries, deterministic timing.

---

## Part 5: F# Implementation for Fidelity

### Core XOR Function

The XOR operation must be expressed in a way that Fidelity can lower efficiently to MLIR. We define a simple, transparent function:

```fsharp
/// XOR two bytes - the fundamental combination operation
let inline xorBytes (a: byte) (b: byte) : byte =
    a ^^^ b
```

The `inline` modifier ensures this compiles to a single XOR instruction with no function call overhead.

### Parallel Tree Combiner

The tree structure is expressed explicitly to enable parallel lowering:

```fsharp
/// Combine four entropy bytes using parallel XOR tree
/// Level 1: (a ⊕ b) and (c ⊕ d) computed in parallel
/// Level 2: results combined
let inline combineEntropy (ch0: byte) (ch1: byte) (ch2: byte) (ch3: byte) : byte =
    let left  = xorBytes ch0 ch1   // Level 1, left branch
    let right = xorBytes ch2 ch3   // Level 1, right branch (parallel)
    xorBytes left right             // Level 2, combine
```

In MLIR, the two Level-1 operations have no data dependency and can be scheduled in parallel or pipelined by the backend.

### ADC Sampling Interface

The ADC read operation is a platform binding—Fidelity's Alex component provides the native implementation:

```fsharp
module Platform.Bindings.ADC =
    /// Read a single sample from the specified ADC channel (0-3)
    /// Returns 10-bit value (0-1023)
    let readChannel (channel: int) : uint16 =
        Unchecked.defaultof<uint16>  // Alex provides implementation
```

### Extracting the Lower 8 Bits

```fsharp
/// Extract lower 8 bits from 10-bit ADC sample
let inline extractByte (sample: uint16) : byte =
    byte (sample &&& 0xFFus)
```

### Complete Entropy Byte Generation

```fsharp
/// Generate one byte of entropy from four ADC channels
let generateEntropyByte () : byte =
    // Sample all four channels
    let s0 = Platform.Bindings.ADC.readChannel 0
    let s1 = Platform.Bindings.ADC.readChannel 1
    let s2 = Platform.Bindings.ADC.readChannel 2
    let s3 = Platform.Bindings.ADC.readChannel 3

    // Extract lower 8 bits from each
    let b0 = extractByte s0
    let b1 = extractByte s1
    let b2 = extractByte s2
    let b3 = extractByte s3

    // Combine using parallel XOR tree
    combineEntropy b0 b1 b2 b3
```

### Expressing Parallelism with scf.parallel

For the sampling phase, we want to express that the four channel reads are logically independent. While the MCP3004 ADC uses a shared SPI bus (preventing true simultaneous reads), expressing the parallelism allows:

1. The compiler to optimize scheduling
2. Future hardware with multiple ADCs to parallelize automatically
3. Documentation of the logical independence in the code structure

```fsharp
/// Generate entropy byte with explicit parallel sampling
/// This maps to MLIR scf.parallel for the read operations
let generateEntropyByteParallel () : byte =
    // Parallel region: four independent reads
    // MLIR: scf.parallel (%ch) = (%c0) to (%c4) step (%c1)
    let samples = Array.Parallel.init 4 (fun ch ->
        Platform.Bindings.ADC.readChannel ch
        |> extractByte
    )

    // Reduction: XOR tree
    combineEntropy samples.[0] samples.[1] samples.[2] samples.[3]
```

### Bulk Generation for 4096 Bytes

```fsharp
/// Generate the full entropy buffer
let generateEntropyBuffer (size: int) : byte array =
    Array.init size (fun _ -> generateEntropyByte ())
```

For the target 4096 bytes:

```fsharp
let quantumEntropy = generateEntropyBuffer 4096
```

---

## Part 6: MLIR Lowering

### Expected MLIR Structure

The `combineEntropy` function should lower to MLIR operations in the `arith` dialect:

```mlir
// combineEntropy lowered to MLIR
func.func @combineEntropy(%ch0: i8, %ch1: i8, %ch2: i8, %ch3: i8) -> i8 {
    // Level 1: parallel XORs (no dependency between these)
    %left = arith.xori %ch0, %ch1 : i8
    %right = arith.xori %ch2, %ch3 : i8

    // Level 2: combine
    %result = arith.xori %left, %right : i8

    return %result : i8
}
```

### Parallel Sampling Region

The four-channel sampling can be expressed with `scf.parallel` and a reduction:

```mlir
func.func @generateEntropyByte() -> i8 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %init = arith.constant 0 : i8
    %mask = arith.constant 255 : i32

    // Parallel sampling with XOR reduction
    %result = scf.parallel (%ch) = (%c0) to (%c4) step (%c1)
              init(%init) -> i8 {

        // Read ADC channel
        %sample = func.call @adc_read_channel(%ch) : (index) -> i32

        // Extract lower 8 bits
        %masked = arith.andi %sample, %mask : i32
        %byte = arith.trunci %masked : i32 to i8

        // XOR reduction
        scf.reduce(%byte : i8) {
        ^bb0(%a: i8, %b: i8):
            %xor = arith.xori %a, %b : i8
            scf.reduce.return %xor : i8
        }
    }

    return %result : i8
}
```

The `scf.parallel` with `scf.reduce` expresses:
- Four logically parallel iterations
- XOR as the associative reduction operator
- The compiler determines actual parallelism based on target hardware

---

## Part 7: Performance Analysis

### Timing Breakdown

For the YoshiPi with MCP3004 over SPI:

| Operation | Time (estimated) |
|-----------|------------------|
| SPI transaction (one channel) | ~25 µs |
| Four channel reads (serial on shared bus) | ~100 µs |
| XOR operations (native code) | ~0.01 µs |
| **Total per byte** | **~100 µs** |

For 4096 bytes:
```
4096 × 100 µs = 409,600 µs ≈ 410 ms
```

### Why This Is Fast Despite Bus Constraints

**No wasted samples**: Von Neumann debiasing discards 50-75% of reads. Our method uses every sample.

**Deterministic timing**: No retry loops, no variable-length sequences. Exactly 4096 × 4 = 16,384 ADC reads for 4096 bytes.

**Minimal computation**: XOR is a single CPU instruction. The tree depth is only 2 operations regardless of data values.

**Comparison with Python baseline**:

| Method | Time for 4096 bytes | Samples used |
|--------|---------------------|--------------|
| Python + Von Neumann | 15,200 ms | ~50,000 (with discards) |
| Python + LSB extraction | 1,117 ms | ~8,200 |
| Native + 4-channel XOR | ~410 ms | 16,384 (deterministic) |

The native parallel XOR approach achieves the <500 ms target while providing stronger mathematical guarantees about entropy quality.

### I2C Consideration

Chris Tacke noted that I2C ADC access on the YoshiPi is significantly slower than SPI. If using I2C:

| Bus Type | Transaction Time | 4096 Bytes |
|----------|------------------|------------|
| SPI (1 MHz) | ~25 µs | ~410 ms |
| I2C (100 kHz) | ~200 µs | ~3,200 ms |
| I2C (400 kHz) | ~50 µs | ~820 ms |

Even with I2C at standard speed, the method remains practical—3.2 seconds is long but acceptable for one-time credential generation. The mathematical soundness is preserved regardless of bus speed.

---

## Part 8: Security Considerations

### Why Four Sources Provide Defense in Depth

The XOR combination provides resilience against partial failures:

- **One channel fails** (stuck or predictable): Three other sources still provide entropy; output remains random
- **Two channels fail**: Two sources still XOR'd; bias reduction is ε² instead of ε⁴
- **Three channels fail**: Single source remains; output has original bias ε

Complete failure requires all four independent quantum processes to be compromised simultaneously—an extremely unlikely scenario for properly isolated circuits.

### No Algorithmic Trust Required

Unlike whitening with SHA-256 or AES, XOR combination:
- Has no hidden state
- Cannot be backdoored
- Is trivially auditable (the operation is visible in a single instruction)
- Provides provable bias reduction based on elementary probability

### Verifiable Randomness

The output can be validated with standard statistical tests (NIST SP 800-22, Dieharder) without knowing or trusting the implementation details. The math guarantees quality; testing confirms it.

---

## Summary

The four-channel parallel XOR method provides:

1. **Mathematical soundness**: ε⁴ bias reduction with proof from first principles
2. **Efficiency**: One output byte per sample cycle, no discards
3. **Parallelism**: Tree structure enables concurrent execution
4. **Simplicity**: Three XOR operations per byte, no complex algorithms
5. **Transparency**: Auditable at every level from physics to machine code
6. **Performance**: Achieves <500 ms target for 4096 bytes on constrained hardware

The F# implementation lowers cleanly through Fidelity to MLIR, preserving the parallel structure for optimization while generating efficient native code for the target platform.

---

## Appendix: Quick Reference

### The Back-Pocket Explanation

> "Each of my four avalanche circuits might have slight bias—maybe 55/45 instead of perfect 50/50. That's a 5% deviation, which we call epsilon. When I XOR two independent sources, the bias squares. When I XOR four sources in a parallel tree, the bias goes to the fourth power. So my 5% bias becomes 0.005%—a thousand times better. That's why four independent quantum sources combined with XOR gives cryptographic-quality randomness even if each individual source isn't perfect."

### The Tree Diagram

```
     CH0 ──┐
           ⊕ ──┐
     CH1 ──┘   │
               ⊕ ──→ output (8ε⁴ bias)
     CH2 ──┐   │
           ⊕ ──┘
     CH3 ──┘
```

### The Core F# Code

```fsharp
let inline combineEntropy (ch0: byte) (ch1: byte) (ch2: byte) (ch3: byte) : byte =
    let left  = ch0 ^^^ ch1   // Level 1, left branch
    let right = ch2 ^^^ ch3   // Level 1, right branch
    left ^^^ right            // Level 2, combine
```
