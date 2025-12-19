# JellyJar: Solution Coherence for Fidelity

## The Metaphor

If you grew up catching fireflies on summer evenings, you may remember putting them in a jar and watching something remarkable happen. At first, each firefly flashes independently, on its own rhythm. But after a few minutes in the jar together, they begin to synchronize. Their flashing aligns. They pulse in unison.

Then you release them. They disperse across the yard, flying off to their own destinations. But for a while, they keep flashing together. They've internalized the shared rhythm. Even as independent agents, they remain in sync.

That's JellyJar.

In the Fidelity framework, independent compilation targets (a microcontroller firmware, a Linux service, a Cloudflare Worker, a desktop application) need to communicate. They share data structures. They pass messages. They honor contracts. JellyJar is the mechanism that brings these independent projects together, synchronizes their understanding of shared interfaces, and then releases them to compile independently while remaining compatible.

## Why .NET Solutions Aren't Enough

If you've worked with .NET, you know the `.sln` file. A solution groups related projects. You can build them together, share references, and Visual Studio understands the relationships.

But a .NET solution is relatively shallow. It's a container. Projects within a solution can reference each other, but the solution itself doesn't enforce much. You can have incompatible versions, mismatched interfaces, or runtime type mismatches that only surface when the code actually runs.

Fidelity targets a different world. When you compile F# to native code without a runtime, there's no dynamic type checking. There's no reflection to discover interfaces at runtime. If two components disagree about the shape of a message, you don't get an exception; you get memory corruption or silent data loss.

This is especially acute for distributed applications. Consider a system with an embedded sensor (ARM Cortex-M), an edge aggregator (Linux x86_64), and a cloud processor (Cloudflare Workers via WebAssembly). All three need to agree on the binary layout of sensor readings. All three need to serialize and deserialize identically. There's no common runtime to mediate.

JellyJar ensures they agree at compile time.

## What JellyJar Manages

### The Program Hypergraph

A single Fidelity project produces a Program Semantic Graph (PSG): the complete semantic structure of that compilation unit. JellyJar manages something larger: a Program Hypergraph that spans multiple projects.

The hypergraph captures:

- **Shared type definitions**: Records, discriminated unions, and structs that appear in multiple projects
- **BAREWire contracts**: Binary encoding specifications that define wire formats
- **Platform binding surfaces**: Which platform bindings each project uses
- **Cross-project references**: Function calls, type references, and data flows across project boundaries
- **Target specifications**: The platform, architecture, and output format for each project

When you build a JellyJar solution, the hypergraph is constructed first. JellyJar analyzes the relationships, detects inconsistencies, and ensures all projects share compatible definitions before any individual compilation begins.

### BAREWire Contract Enforcement

BAREWire provides zero-copy binary encoding for Fidelity. A BAREWire schema defines exact byte layouts: field order, alignment, endianness, and serialization rules.

In a distributed system, multiple components must agree on these layouts. If the sensor firmware encodes a timestamp as a 64-bit little-endian integer at offset 8, the cloud processor had better read it the same way.

JellyJar tracks BAREWire contracts across the solution. It verifies that every project using a particular message type agrees on its encoding. If the edge aggregator references `SensorReading` and the firmware defines it, JellyJar confirms they have identical BAREWire representations.

This happens at solution build time, before any individual project compiles. A mismatch produces an error, not a runtime bug on a deployed device.

### Cross-Target Type Compatibility

Different compilation targets have different type representations. An `int` on a 32-bit microcontroller might be 32 bits; on a 64-bit Linux system, it's still 32 bits in F#, but pointer sizes differ. A `nativeint` varies by platform.

JellyJar tracks these differences. When a shared type flows across platform boundaries, JellyJar verifies that the representation is portable. If you're passing a structure containing `nativeint` between an ARM32 and x86_64 target, JellyJar flags this as a potential problem.

For types that must be portable, JellyJar can enforce BAREWire encoding at the boundary. The internal representation may vary; the wire format is fixed.

## Solution Structure

A JellyJar solution might use a `.fidsln` file (Fidelity Solution) that defines the constituent projects and their relationships:

```toml
[solution]
name = "DistributedSensorNetwork"

[projects]
firmware = { path = "./firmware/sensor.fidproj", target = "arm-cortex-m4" }
edge = { path = "./edge/aggregator.fidproj", target = "linux-x86_64" }
cloud = { path = "./cloud/processor.fidproj", target = "wasm32" }

[contracts]
sensor_protocol = { schema = "./schemas/sensor.bare", used_by = ["firmware", "edge"] }
aggregation_protocol = { schema = "./schemas/aggregation.bare", used_by = ["edge", "cloud"] }

[constraints]
# All projects must agree on these types
shared_types = ["SensorReading", "AggregatedBatch", "ProcessingResult"]
```

The `[contracts]` section explicitly lists BAREWire schemas and which projects use them. JellyJar enforces that these projects have compatible implementations.

The `[constraints]` section lists types that must be identical across specified projects. JellyJar verifies structural equality, not just name equality.

## External Contract Participants

Not every participant in a distributed system is a Fidelity project. A Cloudflare Worker written in Rust might consume messages from a Fidelity edge service. A legacy C application might produce data that Fidelity firmware reads.

JellyJar can track these external participants through contract specifications:

```toml
[external]
legacy_sensor = { schema = "./schemas/legacy.bare", role = "producer" }
rust_worker = { schema = "./schemas/aggregation.bare", role = "consumer" }
```

JellyJar doesn't compile these external components, but it verifies that Fidelity projects expecting to communicate with them use compatible schemas. The BAREWire schema serves as the contract; JellyJar ensures the Fidelity side honors it.

## The Synchronization Process

When you build a JellyJar solution, the process proceeds in phases:

**Phase 1: Discovery**

JellyJar reads all `.fidproj` files in the solution. It identifies shared types, BAREWire schemas, and cross-project references. This produces the initial hypergraph structure.

**Phase 2: Contract Resolution**

For each BAREWire contract, JellyJar locates all projects that reference it. It computes the expected binary representation based on the schema and the types each project defines.

**Phase 3: Compatibility Verification**

JellyJar compares representations across projects. Structural mismatches (different field order, incompatible types, misaligned offsets) produce errors with detailed diagnostics.

**Phase 4: Individual Compilation**

Once coherence is verified, each project compiles independently via Firefly. The PSG for each project is built knowing that its shared types are compatible with sibling projects.

**Phase 5: Artifact Correlation**

After compilation, JellyJar can produce a manifest correlating the binaries. This manifest might record which binary corresponds to which target, which schemas are embedded, and how to deploy the complete system.

## Relationship to Other Fidelity Components

JellyJar sits above Firefly in the tooling hierarchy:

| Component | Role |
|-----------|------|
| **Firefly** | Compiles a single `.fidproj` to a native binary |
| **JellyJar** | Coordinates multiple projects, ensures coherence, invokes Firefly for each |
| **BAREWire** | Defines binary encoding schemas that JellyJar enforces |
| **FSNAC** | Provides IDE support; could integrate with JellyJar for solution-wide diagnostics |

For single-project workflows, you use Firefly directly. For multi-project distributed systems, JellyJar orchestrates the build.

## When You Need JellyJar

JellyJar becomes valuable when:

- **Multiple targets share data**: An embedded device and a host application exchange binary messages
- **Distributed systems**: Microservices, edge computing, or IoT deployments where components run on different platforms
- **Mixed Fidelity/external**: Some components are Fidelity projects, others are Rust, C, or other languages, but all must agree on wire formats
- **Long-lived contracts**: The binary format must remain stable across versions for backward compatibility

If your project is a single binary for a single platform, Firefly alone suffices. The moment you have two components that need to agree on something, JellyJar helps ensure they actually do.

## Comparison to .NET Solutions

| Aspect | .NET Solution | JellyJar Solution |
|--------|---------------|-------------------|
| Project grouping | Yes | Yes |
| Shared references | Assembly references | Type and contract references |
| Cross-project type checking | Via shared assemblies | Via hypergraph verification |
| Binary compatibility | Runtime type checking | Compile-time BAREWire verification |
| Multi-target | Limited (portable libraries) | Native: each project can target different platform |
| External contract enforcement | None | BAREWire schema validation |
| Memory layout verification | None | Explicit layout checking |

## Implementation Status

JellyJar is in the design phase. The concepts described here represent our current thinking about how solution-level coherence should work in the Fidelity ecosystem.

The immediate priorities are:

1. Define the `.fidsln` format and project relationship model
2. Implement hypergraph construction from multiple `.fidproj` files
3. Add BAREWire schema comparison and compatibility checking
4. Integrate with Firefly's build orchestration

As Fidelity matures and more projects require multi-target compilation, JellyJar will become essential infrastructure.

## The Release

The fireflies eventually leave the jar. Each project compiles to its own binary, deploys to its own platform, runs its own code. But they carry with them the shared rhythm: the compatible types, the agreed-upon wire formats, the synchronized understanding of how to communicate.

JellyJar's job is done when the builds complete successfully. The coherence it established persists in the binaries themselves.
