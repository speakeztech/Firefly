# Alex: A Hypergraph-Native MLIR Generation Architecture for Firefly

## Executive Summary

Alex represents the transformation engine within the Firefly compiler that bridges the gap between F#'s rich semantic model and MLIR's powerful compiler infrastructure. This architectural memo documents the design decisions, implementation strategies, and external influences that shape Alex's approach to generating efficient native code for CPU targets.

The core innovation of Alex lies in its preservation of semantics throughout compilation, enabling minimal initial and pass/transform generation of optimized MLIR code. By synthesizing insights from the mlir-hs project's functional binding approach and triton-cpu's systematic lowering patterns, while leveraging functional C++ and TableGen for delimited continuation support, Alex achieves a unique position in the compiler design space.

## Architectural Context

### The Compilation Challenge

Traditional compilation pipelines suffer from what we term "semantic decomposition loss" - the progressive degradation of high-level program structure through multiple intermediate representations. Each transformation stage must reconstruct information that was explicit in earlier representations, leading to both inefficiency and missed optimization opportunities.

The Firefly compiler will eventually address this through its Program Hypergraph (PHG) representation, which will preserve multi-way relationships that traditional graph representations often lose. Alex serves as the bridge from this rich multi-dimensional semantic representation to MLIR's dialect ecosystem, without losing the structural insights that enable optimal code generation.

### Design Principles

1. **Hypergraph Preservation**: In the future state, the PHG preserves multi-way relationships in the source program remain visible through MLIR generation
2. **Single-Pass Intelligence**: Generate optimal MLIR and nano-pass transforms directly rather than relying on multiple restructuring passes
3. **Temporal Learning**: Leverage compilation history to improve code generation over time
4. **Gradient-Based Targeting**: Observe that code exists on a spectrum between control-flow and data-flow paradigms and representation can flow between them if full semantics are preserved
5. **Functional Composition**: Use functional C++, C/k algorithms and TableGen patterns to ensure correctness by construction

## External Influences and Synthesis

### mlir-hs: Functional Bindings as Design Pattern

The mlir-hs project demonstrates how to create type-safe bindings between a functional language (Haskell) and MLIR's C++ infrastructure. While Firefly doesn't require language bindings (being implemented in F# with native MLIR integration), mlir-hs provides crucial insights into functional abstraction patterns over MLIR transform operations.

#### Key Lessons from mlir-hs

**TableGen for Type-Safe Generation**: The mlir-hs approach of using TableGen to generate Haskell bindings (`tblgen/hs-generators.cc`) demonstrates how to maintain type safety across language boundaries. This is a critical insight even though Firefly has a different approach. Alex adapts this pattern for generating F# combinators that construct MLIR operations:

```cpp
// mlir-hs pattern (from hs-generators.cc)
class SimpleAttrPattern : public AttrPattern {
  SimpleAttrPattern(const AttrPatternTemplate& tmpl, NameSource& gen)
    : _type_var_defaults(tmpl.type_var_defaults) {
    // ... generates Haskell type-safe patterns
  }
};

// Alex adaptation: Generate F# combinators instead
class HyperedgePattern : public MLIRPattern {
  HyperedgePattern(const PHGTemplate& tmpl, SemanticContext& ctx)
    : semantic_preserving(true) {
    // Generates F# combinators that preserve hyperedge semantics
    // Key difference: We're generating combinators, not bindings
  }
};
```

**AST as Explicit Representation**: mlir-hs maintains an explicit AST (`src/MLIR/AST.hs`) that mirrors MLIR's type system in Haskell. This approach parallels Alex's preservation of F# type information through the compilation pipeline:

```haskell
-- mlir-hs AST representation
data Type =
    IntegerType Signedness UInt
  | FunctionType [Type] [Type]
  | MemRefType { memrefTypeShape :: [Maybe Int]
               , memrefTypeElement :: Type }
```

Alex will extend this concept by future state hypergraph designs that preserve semantic information:

```fsharp
// Alex hypergraph-aware type representation
type HypergraphType =
    | SimpleType of MLIRType
    | HyperedgeType of {
        Participants: Set<NodeId>
        Relationship: SemanticRelation
        MLIRMapping: MLIRType list
    }
```

### triton-cpu: Systematic Lowering Patterns

The triton-cpu project provides a masterclass in systematic dialect lowering, demonstrating how to transform high-level operations through progressive refinement while maintaining semantic correctness.

#### Key Lessons from triton-cpu

**Two-Stage Conversion Architecture**: Triton-cpu's approach of converting through an intermediate dialect (`TritonToTritonCPU/ConvertDotOp.cpp`) before lowering to LLVM demonstrates the value of domain-specific intermediate representations:

```cpp
// triton-cpu pattern
struct DotOpConversion : public OpConversionPattern<triton::DotOp> {
  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Converts Triton DotOp to TritonCPU DotOp
    rewriter.replaceOpWithNewOp<cpu::DotOp>(op, a, b, c,
                                            op.getInputPrecision(),
                                            op.getMaxNumImpreciseAcc());
  }
};
```

Alex was designed from the beginning to de-compose and re-compose semantic structure in a way that allowed both preservation and where appropriate transformation of programmatic intent. The motivation with starting from a Program Semantic Graph instead of a "raw" symbolic AST is to synthesize the intermediate dialect in a way that preserves types, memory mapping and functional integrity by generating target-appropriate MLIR directly from the PSG/PHG graph:

```cpp
// Alex pattern: Direct semantic-preserving lowering
struct HyperedgeMatMulConversion : public SemanticPattern<PHGMatMul> {
  MLIROperation* generate(const PHGMatMul& matmul,
                         const CompilationContext& ctx) override {
    // Analyze hyperedge structure
    auto gradient = ctx.computeGradient(matmul);

    if (gradient.isDataflow()) {
      // Generate spatial kernel for dataflow architectures
      return generateSpatialMatMul(matmul);
    } else {
      // Generate cache-friendly loops for von Neumann
      return generateTiledMatMul(matmul);
    }
  }
};
```

**Memory Operation Patterns**: The `TritonCPUToLLVM/MemoryOpToLLVM.cpp` implementation shows how to handle complex memory structures through LLVM struct manipulations. This may serve to influence Alex's approach to F# record and discriminated union compilation, though our BAREWire implementation has specific type-preserving requirements that may supersede this implementation. It should serve as a useful example relative to its intended scope:

```cpp
// triton-cpu memory handling
struct ExtractMemRefOpConversion : public OpConversionPattern<ExtractMemRefOp> {
  LogicalResult matchAndRewrite(ExtractMemRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Explicit struct manipulation for memory references
    Value res = b.undef(memRefStructTy);
    res = copyValue(res, 0, 1);  // Copy base
    res = rewriter.create<LLVM::InsertValueOp>(loc, memRefStructTy, res,
                                               b.i64_val(0), 2); // Zero offset
  }
};
```

## Functional C++ and Delimited Continuations

### The Functional C++ Paradigm

Alex intends to employ functional C++ patterns to structure many of its nanopass transforms. This applies to C/k uses for preserving delimited continuation as well as eventually lowering to the DCont dialect directly. This approach, inspired by both projects but extended for Firefly, intends to use immutable transformations and explicit effect tracking:

```cpp
// Functional transformation pattern in Alex
template<typename T>
class ImmutableTransform {
  const T apply(const PHGNode& node, const Context& ctx) const {
    // No mutation - returns new structure
    return T::create(
      transform_impl(node),
      ctx.withNewBindings(node.bindings)
    );
  }

protected:
  virtual T transform_impl(const PHGNode&) const = 0;
};

// Delimited continuation support
class ContinuationBuilder : public ImmutableTransform<MLIRFunc> {
  MLIRFunc transform_impl(const PHGNode& node) const override {
    if (node.hasAsyncBoundary()) {
      return buildDelimitedContinuation(node);
    }
    return buildDirectFunction(node);
  }

private:
  MLIRFunc buildDelimitedContinuation(const PHGNode& node) const {
    // Generate continuation-passing style
    auto [setup, suspension, resumption] =
      splitAtSuspensionPoints(node);

    return MLIRFunc::createCPS(
      setup.generateMLIR(),
      suspension.asContinuation(),
      resumption.asCallback()
    );
  }
};
```

### TableGen Integration for Pattern Matching

TableGen serves as the bridge between high-level patterns and low-level transformations. Unlike mlir-hs which uses TableGen for language bindings, or triton-cpu which uses it for dialect definitions, Alex employs TableGen for semantic pattern matching:

```tablegen
// Alex TableGen patterns for hypergraph-aware compilation
def AsyncMergePattern : HypergraphPattern<
  (PHG_AsyncMerge $streams, $merger, $continuation),
  [{
    // Recognize multi-way async merge as hyperedge
    auto hyperedge = ctx.createHyperedge($streams);

    // Generate appropriate MLIR based on gradient
    if (ctx.gradient.isDataflow()) {
      return generateStreamingMerge(hyperedge, $merger);
    } else {
      return generateStateMachineMerge(hyperedge, $merger);
    }
  }]
>;

def DiscriminatedUnionPattern : SemanticPattern<
  (PHG_DU $discriminator, $cases),
  [{
    // F# DU to MLIR lowering
    auto structType = buildUnionStruct($cases);
    auto switchOp = buildPatternMatch($discriminator, $cases);
    return combineIntoMLIR(structType, switchOp);
  }]
>;
```

## Implementation Architecture

### Core Components

#### 1. Hypergraph Analyzer

While early efforts with Alex will work with the current PSG, a future analyzer may examine a Program Hypergraph to identify patterns and compute architectural gradients:

```cpp
class HypergraphAnalyzer {
  struct AnalysisResult {
    float dataflowGradient;      // 0.0 = pure control, 1.0 = pure dataflow
    Set<HyperedgePattern> patterns;
    Map<NodeId, CoeffectSet> coeffects;
    TemporalHints history;
  };

  AnalysisResult analyze(const ProgramHypergraph& phg) {
    // Identify hyperedges (multi-way relationships)
    auto hyperedges = extractHyperedges(phg);

    // Compute gradient based on hyperedge types
    float gradient = computeGradient(hyperedges);

    // Extract patterns for TableGen matching
    auto patterns = recognizePatterns(hyperedges);

    // Gather temporal hints from compilation database
    auto history = queryTemporalDatabase(phg.fingerprint);

    return {gradient, patterns, coeffects, history};
  }
};
```

#### 2. MLIR Generator

The generator produces MLIR operations directly from hypergraph patterns:

```cpp
class MLIRGenerator {
  MLIRModule generate(const ProgramHypergraph& phg,
                     const AnalysisResult& analysis) {
    MLIRModule module;

    // Process hyperedges in parallel when possible
    parallel_for(phg.hyperedges, [&](const Hyperedge& edge) {
      // Each hyperedge generates a coherent MLIR fragment
      auto mlirOps = generateFromHyperedge(edge, analysis);
      module.addOperations(mlirOps);
    });

    return module;
  }

private:
  vector<MLIROperation*> generateFromHyperedge(
      const Hyperedge& edge,
      const AnalysisResult& analysis) {
    // Apply TableGen patterns
    if (auto pattern = matchPattern(edge)) {
      return pattern->generate(edge, analysis);
    }

    // Fall back to direct generation
    return directGeneration(edge);
  }
};
```

#### 3. Continuation Transformer

Handles delimited continuations for async operations:

```cpp
class ContinuationTransformer {
  struct ContinuationPoint {
    MLIRBlock* suspensionBlock;
    MLIRValue* state;
    MLIRFunc* resumption;
  };

  void transformAsync(PHGAsyncNode& async, MLIRModule& module) {
    // Split at suspension points
    auto continuations = identifyContinuationPoints(async);

    // Generate state machine if needed
    if (continuations.size() > 1) {
      generateStateMachine(continuations, module);
    } else {
      // Simple CPS transformation
      generateCPS(continuations[0], module);
    }
  }

  void generateStateMachine(
      const vector<ContinuationPoint>& points,
      MLIRModule& module) {
    // Create state enum
    auto stateType = module.addEnumType(points.size());

    // Generate dispatcher
    auto dispatcher = module.addFunc("dispatch");
    auto switchOp = dispatcher.addSwitch(stateType);

    // Each continuation becomes a case
    for (auto& point : points) {
      auto caseBlock = switchOp.addCase(point.suspensionBlock);
      caseBlock.addCall(point.resumption);
    }
  }
};
```

### XParsec Integration for Combinator-Based Generation

Alex uses XParsec-style combinators for composing MLIR generation patterns:

```fsharp
// F# combinators for MLIR generation
module AlexCombinators =
    open XParsec

    // Basic MLIR builders
    let func name params body =
        mlir {
            let! f = MLIRFunc.create name params
            let! entry = f.addEntryBlock()
            let! result = body entry
            do! f.setReturn result
            return f
        }

    // Hyperedge-aware combinators
    let hyperedge participants coordinator =
        mlir {
            // Preserve multi-way relationship
            let! edges = many participants
            let! hub = coordinator edges
            return HyperedgeMLIR(edges, hub)
        }

    // Temporal combinators
    let withHistory pattern =
        mlir {
            let! fingerprint = computeFingerprint pattern
            let! history = queryHistory fingerprint
            match history with
            | Some(prev, performance) when performance > 0.8 ->
                return! reuseStrategy prev
            | _ ->
                return! generateFresh pattern
        }
```

## Compilation Pipeline Integration

### Phase 2: Analysis

Analyze the PSG/PHG to determine compilation strategy:

```cpp
CompilationStrategy analyzeHypergraph(const ProgramHypergraph& phg) {
  HypergraphAnalyzer analyzer;
  auto result = analyzer.analyze(phg);

  CompilationStrategy strategy;

  // Determine overall approach based on gradient
  if (result.dataflowGradient > 0.7) {
    strategy.approach = CompilationApproach::DataflowOriented;
    strategy.optimizations.push_back(StreamFusion);
    strategy.optimizations.push_back(PipelineParallelism);
  } else if (result.dataflowGradient < 0.3) {
    strategy.approach = CompilationApproach::ControlFlowOriented;
    strategy.optimizations.push_back(LoopTiling);
    strategy.optimizations.push_back(CacheOptimization);
  } else {
    strategy.approach = CompilationApproach::Hybrid;
    strategy.optimizations.push_back(SelectiveVectorization);
  }

  // Apply temporal learning
  if (result.history.hasSuccessfulCompilations()) {
    strategy.hints = result.history.bestStrategy();
  }

  return strategy;
}
```

### Phase 2: MLIR Generation

Generate MLIR directly from the hypergraph:

```cpp
MLIRModule generateMLIR(const ProgramHypergraph& phg,
                       const CompilationStrategy& strategy) {
  MLIRGenerator generator(strategy);
  ContinuationTransformer continuations;

  // Generate module structure
  auto module = generator.createModule(phg.metadata);

  // Process each hyperedge
  for (const auto& edge : phg.hyperedges) {
    // Apply TableGen patterns
    if (auto ops = applyPatterns(edge, strategy)) {
      module.addOperations(ops);
    }

    // Handle continuations
    if (edge.hasContinuations()) {
      continuations.transform(edge, module);
    }
  }

  // Single-pass optimization based on strategy
  applyStrategicOptimizations(module, strategy);

  return module;
}
```

It's important to recognize that there are two generative passes - first is the "top level" MLIR that is the direct expression of the PSG/PHG according to Alex preliminary analysis. The *second* pass is *also* generated here which is the transforms that will take that initially generated MLIR dialect information to them lower it to the appropriate LLVM IR expressions that will then be compiled in the "back end" to the native binary.

### Phase 3: LLVM Lowering

The is the "back end" in this early scenario. Eventually Alex will be tasked with transforming top-level MLIR to a variety of "back ends" for compilation to other target processors. But in this case we are focused on standard command-line based compilation for desktop applications. Therefore LLVM is "the world" as far as Alex is concerned at this stage of platform development.

Here the MLIR is transformed to LLVM IR:

```cpp
LLVMModule lowerToLLVM(const MLIRModule& mlir,
                      const TargetInfo& target) {
  // Create lowering pipeline
  PassManager pm;

  // Add minimal passes - most optimization already done
  pm.addPass(createConvertToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  // Target-specific lowering
  if (target.hasVectorExtensions()) {
    pm.addPass(createVectorToLLVMPass());
  }

  // Run pipeline
  pm.run(mlir);

  // Translate to LLVM IR
  return translateToLLVMIR(mlir);
}
```

This is sample pseudo-code, and may take the form of code that's executed by opt_mlir to verify and then handed off to LLVM to perform its tasks of LTO and final lowering to bytecode and then a native application.

## Performance Characteristics

### Nano-Pass Efficiency

By preserving hypergraph structure and generating optimal MLIR directly, Alex eliminates excess MLIR (and LLVM) optimization passes:

**Traditional Pipeline** (triton-cpu style):
- Parse: O(n)
- Type check: O(n)
- Lower to IR₁: O(n)
- Optimize IR₁: O(n²) for some optimizations
- Lower to IR₂: O(n)
- Optimize IR₂: O(n²)
- ... (multiple passes)
- Total: O(kn²) where k is number of passes

**Alex Pipeline**:
- Parse (FCS): O(n)
- Build PHG: O(n)
- Analyze hypergraph: O(n log n) for pattern matching
- Generate optimal MLIR: O(n)
- Lower to LLVM: O(n)
- Total: O(n log n)

### Memory Characteristics

The hypergraph representation requires more memory than traditional ASTs but eliminates intermediate representations:

- PHG memory: ~2x traditional AST
- No intermediate IRs: Saves k × AST size
- Net benefit when k > 2 (typical pipelines have k > 5)

## Example: Compiling HelloWorldDirect

To illustrate the complete pipeline, consider the HelloWorldDirect example:

```fsharp
// F# source
let hello() =
    use buffer = stackBuffer<byte> 256
    Console.Write "Enter your name: "
    let name =
        match Console.readInto buffer with
        | Ok length -> spanToString (buffer.AsReadOnlySpan(0, length))
        | Error _ -> "Unknown Person"
    Console.WriteLine $"Hello, {name}!"
```

### Step 1: PSG Construction (FCS)

```fsharp
// Simplified PSG representation
PSGNode.Let("hello",
  PSGNode.Function([],
    PSGNode.Use("buffer",
      PSGNode.StackBuffer(256),
      PSGNode.Sequence([
        PSGNode.Call("Console.Write", ["Enter your name: "])
        PSGNode.Match(
          PSGNode.Call("Console.readInto", ["buffer"]),
          [
            PSGCase.Ok("length", ...)
            PSGCase.Error(_, ...)
          ])
      ]))))
```

### Step 2: MLIR Generation

```mlir
func @hello() {
  // Stack allocation (zero-heap)
  %buffer = memref.alloca() : memref<256xi8>

  // Console write (external call)
  %prompt = llvm.mlir.constant("Enter your name: ") : !llvm.ptr<i8>
  call @Console_Write(%prompt) : (!llvm.ptr<i8>) -> ()

  // Read with error handling
  %result = call @Console_readInto(%buffer) : (memref<256xi8>) -> i32
  %success = cmpi "sge", %result, %c0 : i32

  // Pattern match via scf.if
  %name = scf.if %success -> !llvm.ptr<i8> {
    %span = memref.subview %buffer[0][%result][1] : memref<256xi8>
    %str = call @spanToString(%span) : (memref<?xi8>) -> !llvm.ptr<i8>
    scf.yield %str : !llvm.ptr<i8>
  } else {
    %default = llvm.mlir.constant("Unknown Person") : !llvm.ptr<i8>
    scf.yield %default : !llvm.ptr<i8>
  }

  // Final output
  call @Console_WriteLine(%name) : (!llvm.ptr<i8>) -> ()

  // Automatic cleanup (stack allocated)
  return
}
```

### Step 3: LLVM IR

```llvm
define void @hello() {
entry:
  %buffer = alloca [256 x i8], align 1
  %prompt = getelementptr inbounds [17 x i8], [17 x i8]* @.str.1, i32 0, i32 0
  call void @Console_Write(i8* %prompt)

  %result = call i32 @Console_readInto(i8* %buffer)
  %success = icmp sge i32 %result, 0
  br i1 %success, label %then, label %else

then:
  %str = call i8* @spanToString(i8* %buffer, i32 %result)
  br label %continue

else:
  br label %continue

continue:
  %name = phi i8* [ %str, %then ], [ getelementptr inbounds ([15 x i8], [15 x i8]* @.str.2, i32 0, i32 0), %else ]
  call void @Console_WriteLine(i8* %name)
  ret void
}
```

## Future Directions

### Temporal Learning Implementation

The compilation database will track patterns and performance:

```sql
CREATE TABLE compilation_patterns (
    pattern_hash VARCHAR PRIMARY KEY,
    hyperedge_type VARCHAR,
    mlir_generated TEXT,
    performance_metrics JSONB,
    target_architecture VARCHAR,
    success_rate FLOAT,
    last_used TIMESTAMP
);

CREATE INDEX idx_performance ON compilation_patterns(success_rate);
CREATE INDEX idx_hyperedge ON compilation_patterns(hyperedge_type);
```

### Heterogeneous Target Support

While the initial focus is CPU compilation, the gradient-based approach naturally extends to heterogeneous systems:

```cpp
class HeterogeneousStrategy : public CompilationStrategy {
  MLIRModule generate(const ProgramHypergraph& phg) override {
    auto gradient = computeGradient(phg);

    MLIRModule cpu_module, gpu_module;

    for (const auto& edge : phg.hyperedges) {
      if (gradient[edge] > 0.7) {
        // Data-parallel -> GPU
        gpu_module.add(generateGPUKernel(edge));
      } else {
        // Control-heavy -> CPU
        cpu_module.add(generateCPUCode(edge));
      }
    }

    return combineModules(cpu_module, gpu_module);
  }
};
```

## Conclusion

Alex represents a synthesis of functional programming principles, semantic and eventually hypergraph theory, and practical compiler engineering. By learning from mlir-hs's functional abstraction patterns and triton-cpu's systematic lowering approach, while innovating in hypergraph preservation and nano-pass optimization, Alex provides a solid foundation for the Firefly compiler's native code generation.

The key insights that drive this architecture:

1. **Programs are hypergraphs** - Multi-way relationships are fundamental, not decomposable and should be preserved for efficiency and optimization at higher abstractions
2. **Semantic preservation enables optimization** - Don't lose information then struggle to recover it
3. **Functional patterns ensure correctness** - Immutable transformations and explicit effects
4. **Learning improves compilation** - Temporal patterns guide future compilations
5. **Gradients beat binary choices** - Code exists on a spectrum, not in either/or null sum choices

This architecture provides a clear path from F# source code to efficient native binaries, maintaining the semantic richness of the source language while leveraging MLIR's powerful optimization infrastructure. The initial CPU-focused implementation establishes patterns that will naturally extend to heterogeneous and specialized architectures as the Fidelity framework evolves.

## References

- mlir-hs: [GitHub Repository](https://github.com/google/mlir-hs) - Functional bindings and TableGen patterns
- triton-cpu: [GitHub Repository](https://github.com/triton-lang/triton/tree/main/third_party/cpu) - Systematic lowering and dialect design
- MLIR Documentation: [mlir.llvm.org](https://mlir.llvm.org) - Dialect specifications and lowering patterns
- XParsec: Local repository - Combinator patterns for transformation composition
