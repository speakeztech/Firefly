/// PSG Builder - Re-exports from refactored sub-modules for backwards compatibility
///
/// The builder logic has been refactored into:
///   - Builder/Types.fs          - BuildContext and node creation
///   - Builder/SymbolCorrelation.fs - Symbol correlation strategies
///   - Builder/PatternProcessing.fs - Pattern AST processing
///   - Builder/ExpressionProcessing.fs - Expression AST processing
///   - Builder/BindingProcessing.fs - Binding processing
///   - Builder/DeclarationProcessing.fs - Type/member/module declarations
///   - Builder/Main.fs           - Main entry point and orchestration
namespace Core.PSG

open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types

/// Backwards-compatible module that re-exports from refactored sub-modules
module Builder =
    // Re-export the BuildContext type
    type BuildContext = Core.PSG.Construction.Types.BuildContext

    // Re-export the main entry point
    let buildProgramSemanticGraph
        (checkResults: FSharpCheckProjectResults)
        (parseResults: FSharpParseFileResults[]) : ProgramSemanticGraph =
        Core.PSG.Construction.Main.buildProgramSemanticGraph checkResults parseResults
