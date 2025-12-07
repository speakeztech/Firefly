/// ExpressionEmitter - Pure MLIR text serialization
///
/// This module is TRANSPORT ONLY. It provides helpers for serializing
/// values to MLIR text. NO transformation logic belongs here.
///
/// Transformations happen in:
/// - PSG nanopasses (Core/PSG/Nanopass/)
/// - Alex transforms (Alex/Transforms/) using XParsec patterns
module Alex.Emission.ExpressionEmitter

open Alex.CodeGeneration.MLIRBuilder

// ═══════════════════════════════════════════════════════════════════
// Result Type - used by Alex transforms to communicate emission results
// ═══════════════════════════════════════════════════════════════════

/// Result of emitting an expression to MLIR
type EmitResult =
    | Emitted of Val   // Produced an SSA value
    | Void             // No value (unit-returning expression)
    | Error of string  // Emission failed
