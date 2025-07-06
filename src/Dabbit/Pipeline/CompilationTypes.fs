module Dabbit.Pipeline.CompilationTypes

open System
open FSharp.Compiler.Text

type CompilationPhase =
    | ProjectLoading | FCSProcessing | SymbolCollection | ReachabilityAnalysis
    | IntermediateGeneration | MLIRGeneration | LLVMGeneration | NativeCompilation

type ProgressCallback = CompilationPhase -> string -> unit

type FireflyError =
    | InternalError of phase: string * message: string * stackTrace: string option
    | AllocationDetected of typeName: string * location: range
    | EntryPointNotFound of message: string
    | ZeroAllocationViolation of allocatingFunctions: string[]
    | MLIRGenerationError of message: string
    | LLVMError of message: string
    | NativeCompilationError of message: string

type CompilationStatistics = {
    TotalFiles: int
    TotalSymbols: int
    ReachableSymbols: int
    EliminatedSymbols: int
    CompilationTimeMs: float
}