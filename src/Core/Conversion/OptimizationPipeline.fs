module Core.Conversion.OptimizationPipeline

open System
open System.IO
open Core.XParsec.Foundation

/// Optimization level for LLVM code generation
type OptimizationLevel =
    | None        // -O0
    | Less        // -O1  
    | Default     // -O2
    | Aggressive  // -O3
    | Size        // -Os
    | SizeMin     // -Oz

/// LLVM optimization passes
type OptimizationPass =
    | InliningPass of threshold: int
    | InstCombine of aggressiveness: int
    | Reassociate
    | GVN of enableLoads: bool
    | LICM of promotionEnabled: bool
    | MemoryToReg
    | DeadCodeElim of globalElim: bool
    | SCCP of speculative: bool
    | SimplifyCFG of hoistCommonInsts: bool
    | LoopUnroll of threshold: int
    | ConstantFold
    | AlwaysInline

/// LLVM output representation
type LLVMOutput = {
    LLVMIRText: string
    ModuleName: string
    OptimizationLevel: OptimizationLevel
    Metadata: Map<string, string>
}

/// Optimization state
type OptimizationState = {
    CurrentPass: string
    TransformationCount: int
    RemovedInstructions: string list
    OptimizedInstructions: string list
    SymbolRenaming: Map<string, string>
    OptimizationMetrics: Map<string, int>
    ErrorContext: string list
}

/// Creates optimization pipeline based on level
let createOptimizationPipeline (level: OptimizationLevel) : OptimizationPass list =
    match level with
    | None -> []
    | Less -> [MemoryToReg; SimplifyCFG(false)]
    | Default -> [MemoryToReg; SimplifyCFG(false); InstCombine(1); ConstantFold]
    | Aggressive -> [MemoryToReg; SimplifyCFG(true); InstCombine(2); ConstantFold; DeadCodeElim(false); GVN(true)]
    | Size -> [MemoryToReg; DeadCodeElim(true); ConstantFold]
    | SizeMin -> [MemoryToReg; DeadCodeElim(true)]

/// Convert optimization pass to LLVM opt command line argument
let passToOptArg (pass: OptimizationPass) : string =
    match pass with
    | InliningPass threshold -> sprintf "-inline -inline-threshold=%d" threshold
    | InstCombine _ -> "-instcombine"
    | Reassociate -> "-reassociate"
    | GVN enableLoads -> if enableLoads then "-gvn" else "-gvn -no-loads"
    | LICM promotionEnabled -> if promotionEnabled then "-licm" else "-licm -licm-no-promotion"
    | MemoryToReg -> "-mem2reg"
    | DeadCodeElim globalElim -> if globalElim then "-globaldce" else "-dce"
    | SCCP speculative -> if speculative then "-sccp" else "-sccp -no-speculative"
    | SimplifyCFG hoistCommon -> if hoistCommon then "-simplifycfg" else "-simplifycfg -no-hoist-common-insts"
    | LoopUnroll threshold -> sprintf "-loop-unroll -unroll-threshold=%d" threshold
    | ConstantFold -> "-constprop"
    | AlwaysInline -> "-always-inline"

/// Convert optimization level to LLVM opt flag
let levelToOptFlag (level: OptimizationLevel) : string =
    match level with
    | None -> "-O0"
    | Less -> "-O1"
    | Default -> "-O2"
    | Aggressive -> "-O3"
    | Size -> "-Os"
    | SizeMin -> "-Oz"

/// Main optimization entry point
let optimizeLLVMIR (llvmOutput: LLVMOutput) (passes: OptimizationPass list) : CompilerResult<LLVMOutput> =
    if String.IsNullOrWhiteSpace(llvmOutput.LLVMIRText) then
        CompilerFailure [ConversionError("LLVM optimization", "empty LLVM IR", "optimized LLVM IR", "Input LLVM IR cannot be empty")]
    else
        // For POC, apply basic text-based transformations
        let optimized = 
            passes |> List.fold (fun ir pass ->
                match pass with
                | MemoryToReg ->
                    // Simple simulation: remove alloca followed by immediate store/load
                    ir
                | SimplifyCFG _ ->
                    // Simple simulation: remove unreachable blocks
                    ir
                | _ -> ir
            ) llvmOutput.LLVMIRText
        
        Success { llvmOutput with LLVMIRText = optimized }

/// Apply optimization using external LLVM opt tool
let applyOptimizationWithTool (llvmIR: string) (level: OptimizationLevel) (passes: OptimizationPass list) : CompilerResult<string> =
    if String.IsNullOrWhiteSpace(llvmIR) then
        CompilerFailure [ConversionError("LLVM optimization", "empty input", "optimized LLVM IR", "Input LLVM IR is empty")]
    else
        try
            // Create temporary files
            let tempDir = Path.GetTempPath()
            let inputPath = Path.Combine(tempDir, sprintf "firefly_%s.ll" (Guid.NewGuid().ToString("N")))
            let outputPath = Path.Combine(tempDir, sprintf "firefly_%s_opt.ll" (Guid.NewGuid().ToString("N")))
            
            try
                // Write input LLVM IR
                File.WriteAllText(inputPath, llvmIR)
                
                // Build opt command arguments
                let levelFlag = levelToOptFlag level
                let passFlags = passes |> List.map passToOptArg |> String.concat " "
                let args = sprintf "%s %s %s -o %s" levelFlag passFlags inputPath outputPath
                
                // Run opt (would use Process.Start in real implementation)
                let psi = System.Diagnostics.ProcessStartInfo("opt", args)
                psi.UseShellExecute <- false
                psi.RedirectStandardError <- true
                psi.CreateNoWindow <- true
                
                use proc = System.Diagnostics.Process.Start(psi)
                proc.WaitForExit()
                
                if proc.ExitCode = 0 && File.Exists(outputPath) then
                    let optimizedIR = File.ReadAllText(outputPath)
                    Success optimizedIR
                else
                    let error = proc.StandardError.ReadToEnd()
                    CompilerFailure [ConversionError("LLVM opt", "optimization failed", "optimized LLVM IR", error)]
                    
            finally
                // Clean up temp files
                if File.Exists(inputPath) then File.Delete(inputPath)
                if File.Exists(outputPath) then File.Delete(outputPath)
                
        with ex ->
            CompilerFailure [ConversionError("LLVM optimization", "process execution", "optimized LLVM IR", ex.Message)]

/// Validates that optimized IR maintains zero-allocation guarantees
let validateZeroAllocationGuarantees (llvmIR: string) : CompilerResult<unit> =
    let heapPatterns = [
        "call.*@malloc"
        "call.*@calloc" 
        "call.*@realloc"
        "call.*@new"
        "invoke.*@malloc"
        "call.*@_Znwm"  // C++ new
        "call.*@_Znam"  // C++ new[]
    ]
    
    let violations = 
        heapPatterns
        |> List.filter (fun pattern -> 
            System.Text.RegularExpressions.Regex.IsMatch(llvmIR, pattern))
    
    if violations.IsEmpty then
        Success ()
    else
        CompilerFailure [ConversionError(
            "zero-allocation validation", 
            "optimized LLVM IR", 
            "zero-allocation LLVM IR", 
            sprintf "Found potential heap allocation patterns: %s" (String.concat ", " violations))]

/// Creates optimization pipeline for different build profiles
let createProfileOptimizationPipeline (profile: string) : OptimizationPass list =
    match profile.ToLowerInvariant() with
    | "debug" -> [MemoryToReg]
    | "release" -> createOptimizationPipeline Aggressive
    | "size" -> createOptimizationPipeline Size
    | "embedded" -> createOptimizationPipeline SizeMin
    | _ -> createOptimizationPipeline Default

/// Get optimization statistics from LLVM IR
let getOptimizationStats (originalIR: string) (optimizedIR: string) : Map<string, int> =
    let countPattern pattern text =
        System.Text.RegularExpressions.Regex.Matches(text, pattern).Count
    
    let stats = [
        "allocas", countPattern @"%\w+\s*=\s*alloca" originalIR - countPattern @"%\w+\s*=\s*alloca" optimizedIR
        "loads", countPattern @"%\w+\s*=\s*load" originalIR - countPattern @"%\w+\s*=\s*load" optimizedIR
        "stores", countPattern @"store\s+" originalIR - countPattern @"store\s+" optimizedIR
        "branches", countPattern @"br\s+" originalIR - countPattern @"br\s+" optimizedIR
        "functions", countPattern @"define\s+" originalIR
        "basic_blocks", countPattern @"^\w+:" originalIR - countPattern @"^\w+:" optimizedIR
    ]
    
    stats |> Map.ofList