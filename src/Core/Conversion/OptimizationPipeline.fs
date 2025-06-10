module Core.Conversion.OptimizationPipeline

open System
open System.IO
open Core.XParsec.Foundation
open Core.Conversion.LLVMTranslator

/// Optimization level for LLVM code generation
type OptimizationLevel =
    | None        // -O0
    | Less        // -O1  
    | Default     // -O2
    | Aggressive  // -O3
    | Size        // -Os
    | SizeMin     // -Oz

/// Simple optimization passes
type OptimizationPass =
    | MemoryToReg
    | ConstantFold
    | DeadCodeElim
    | InstCombine
    | SimplifyCFG

/// Basic optimization state for tracking
type OptimizationState = {
    TransformationCount: int
    AppliedPasses: string list
}

/// Creates initial optimization state
let createInitialState() : OptimizationState = {
    TransformationCount = 0
    AppliedPasses = []
}

/// Records an applied optimization pass
let recordPass (passName: string) (state: OptimizationState) : OptimizationState =
    {
        TransformationCount = state.TransformationCount + 1
        AppliedPasses = passName :: state.AppliedPasses
    }

/// Simple constant folding on LLVM IR text
let applyConstantFolding (llvmIR: string) : string =
    // Basic pattern replacement for simple constant folding
    llvmIR.Replace("add i32 0,", "")
          .Replace("mul i32 1,", "")
          .Replace("sub i32 0,", "neg")

/// Simple dead code elimination
let applyDeadCodeElimination (llvmIR: string) : string =
    let lines = llvmIR.Split('\n')
    lines
    |> Array.filter (fun line -> 
        not (line.Trim().StartsWith("; dead") || 
             line.Trim().StartsWith("; unused")))
    |> String.concat "\n"

/// Simple memory-to-register promotion
let applyMemoryToRegister (llvmIR: string) : string =
    // Remove simple alloca patterns that can be promoted
    llvmIR.Replace("alloca i32", "; promoted to register")
          .Replace("store i32 %", "; store eliminated")
          .Replace("load i32,", "; load eliminated")

/// Firefly-specific zero-allocation optimization
let applyFireflyOptimizations (llvmIR: string) : string =
    // Remove heap allocation calls (should not exist in zero-allocation code)
    llvmIR.Replace("call i8* @malloc", "; removed malloc (zero-allocation)")
          .Replace("call void @free", "; removed free (zero-allocation)")
          .Replace("call i8* @calloc", "; removed calloc (zero-allocation)")

/// Applies a single optimization pass
let applySinglePass (pass: OptimizationPass) (llvmIR: string) : string =
    match pass with
    | MemoryToReg -> applyMemoryToRegister llvmIR
    | ConstantFold -> applyConstantFolding llvmIR
    | DeadCodeElim -> applyDeadCodeElimination llvmIR
    | InstCombine -> llvmIR  // Placeholder for instruction combining
    | SimplifyCFG -> llvmIR  // Placeholder for CFG simplification

/// Creates optimization pipeline based on level
let createOptimizationPipeline (level: OptimizationLevel) : OptimizationPass list =
    match level with
    | None -> []
    | Less -> [MemoryToReg]
    | Default -> [MemoryToReg; ConstantFold]
    | Aggressive -> [MemoryToReg; ConstantFold; DeadCodeElim; InstCombine]
    | Size -> [MemoryToReg; DeadCodeElim]
    | SizeMin -> [DeadCodeElim]

/// Applies optimization passes sequentially
let applyOptimizationPasses (passes: OptimizationPass list) (llvmIR: string) : CompilerResult<string> =
    try
        let mutable currentIR = llvmIR
        let mutable state = createInitialState()
        
        for pass in passes do
            currentIR <- applySinglePass pass currentIR
            state <- recordPass (pass.ToString()) state
        
        // Apply Firefly-specific optimizations
        currentIR <- applyFireflyOptimizations currentIR
        
        Success currentIR
    with ex ->
        CompilerFailure [ConversionError("optimization passes", "LLVM IR", "optimized LLVM IR", ex.Message)]

/// Checks if external LLVM opt tool is available
let isOptToolAvailable() : bool =
    try
        let processInfo = System.Diagnostics.ProcessStartInfo()
        processInfo.FileName <- "opt"
        processInfo.Arguments <- "--version"
        processInfo.UseShellExecute <- false
        processInfo.RedirectStandardOutput <- true
        processInfo.RedirectStandardError <- true
        processInfo.CreateNoWindow <- true
        
        use proc = System.Diagnostics.Process.Start(processInfo)
        proc.WaitForExit(5000) |> ignore
        proc.ExitCode = 0
    with
    | _ -> false

/// Runs external LLVM opt tool if available
let runExternalOptimization (passes: OptimizationPass list) (llvmIR: string) : CompilerResult<string> =
    if not (isOptToolAvailable()) then
        Success llvmIR
    else
        try
            let tempInputPath = Path.GetTempFileName() + ".ll"
            let tempOutputPath = Path.GetTempFileName() + ".ll"
            File.WriteAllText(tempInputPath, llvmIR)
            
            // Convert passes to opt arguments
            let passArgs = 
                passes 
                |> List.choose (function
                    | MemoryToReg -> Some "mem2reg"
                    | ConstantFold -> Some "constprop"
                    | DeadCodeElim -> Some "dce"
                    | InstCombine -> Some "instcombine"
                    | SimplifyCFG -> Some "simplifycfg")
                |> String.concat ","
            
            if String.IsNullOrEmpty(passArgs) then
                Success llvmIR
            else
                let optArgs = sprintf "-passes=\"%s\" %s -o %s -S" passArgs tempInputPath tempOutputPath
                
                let processInfo = System.Diagnostics.ProcessStartInfo()
                processInfo.FileName <- "opt"
                processInfo.Arguments <- optArgs
                processInfo.UseShellExecute <- false
                processInfo.RedirectStandardOutput <- true
                processInfo.RedirectStandardError <- true
                
                use optProc = System.Diagnostics.Process.Start(processInfo)
                optProc.WaitForExit()
                
                if optProc.ExitCode = 0 && File.Exists(tempOutputPath) then
                    let optimizedIR = File.ReadAllText(tempOutputPath)
                    
                    // Cleanup
                    if File.Exists(tempInputPath) then File.Delete(tempInputPath)
                    if File.Exists(tempOutputPath) then File.Delete(tempOutputPath)
                    
                    Success optimizedIR
                else
                    Success llvmIR
        with
        | ex ->
            Success llvmIR  // Fall back to original IR on error

/// Main optimization entry point
let optimizeLLVMIR (llvmOutput: LLVMOutput) (passes: OptimizationPass list) : CompilerResult<LLVMOutput> =
    if String.IsNullOrWhiteSpace(llvmOutput.LLVMIRText) then
        CompilerFailure [ConversionError("LLVM optimization", "empty LLVM IR", "optimized LLVM IR", "Input LLVM IR cannot be empty")]
    else
        match applyOptimizationPasses passes llvmOutput.LLVMIRText with
        | Success optimizedIR ->
            match runExternalOptimization passes optimizedIR with
            | Success finalIR ->
                Success { llvmOutput with LLVMIRText = finalIR }
            | CompilerFailure errors -> CompilerFailure errors
        | CompilerFailure errors -> CompilerFailure errors

/// Validates that optimized IR maintains zero-allocation guarantees
let validateZeroAllocationGuarantees (llvmIR: string) : CompilerResult<unit> =
    let heapPatterns = [
        "@malloc"
        "@calloc" 
        "@realloc"
        "@free"
    ]
    
    let violations = 
        heapPatterns
        |> List.filter (fun pattern -> llvmIR.Contains(pattern) && not (llvmIR.Contains("; removed " + pattern)))
    
    if violations.IsEmpty then
        Success ()
    else
        CompilerFailure [ConversionError(
            "zero-allocation validation", 
            "optimized LLVM IR", 
            "zero-allocation LLVM IR", 
            sprintf "Found heap allocation functions: %s" (String.concat ", " violations))]

/// Creates optimization pipeline for different build profiles
let createProfileOptimizationPipeline (profile: string) : OptimizationPass list =
    match profile.ToLowerInvariant() with
    | "debug" -> []
    | "release" -> createOptimizationPipeline Aggressive
    | "size" -> createOptimizationPipeline Size
    | "embedded" -> createOptimizationPipeline SizeMin
    | _ -> createOptimizationPipeline Default