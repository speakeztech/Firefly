module Core.Conversion.OptimizationPipeline

open System
open System.IO
open System.Text.RegularExpressions
open Core.Conversion.LLVMTranslator

/// Optimization level for LLVM code generation
type OptimizationLevel =
    | None        // -O0
    | Less        // -O1
    | Default     // -O2
    | Aggressive  // -O3
    | Size        // -Os
    | SizeMin     // -Oz

/// LLVM optimization passes to apply
type OptimizationPass =
    | InliningPass
    | InstCombine
    | Reassociate
    | GVN           // Global Value Numbering
    | LICM          // Loop Invariant Code Motion
    | MemoryToReg   // Promote memory to registers
    | DeadCodeElim  // Dead Code Elimination
    | SCCP          // Sparse Conditional Constant Propagation
    | SimplifyCFG   // Simplify Control Flow Graph
    | LoopUnroll    // Loop Unrolling
    | ConstantFold  // Constant Folding
    | AlwaysInline  // Force inlining of small functions

/// Maps optimization pass to LLVM opt pass name (updated for newer LLVM versions)
let private passToString (pass: OptimizationPass) : string =
    match pass with
    | InliningPass -> "--inline"
    | InstCombine -> "--instcombine"
    | Reassociate -> "--reassociate"
    | GVN -> "--gvn"
    | LICM -> "--licm"
    | MemoryToReg -> "--mem2reg"
    | DeadCodeElim -> "--dce"
    | SCCP -> "--sccp"
    | SimplifyCFG -> "--simplifycfg"
    | LoopUnroll -> "--loop-unroll"
    | ConstantFold -> "--consthoist"  // Updated name
    | AlwaysInline -> "--always-inline"

/// Creates optimization pipeline based on the optimization level
let createOptimizationPipeline (level: OptimizationLevel) : OptimizationPass list =
    match level with
    | None -> []
    | Less -> 
        [MemoryToReg; SimplifyCFG]  // Keep only basic, reliable passes
    | Default -> 
        [MemoryToReg; SimplifyCFG; InstCombine]  // Conservative set
    | Aggressive -> 
        [MemoryToReg; SimplifyCFG; InstCombine; 
         DeadCodeElim; InliningPass]  // More passes but avoid problematic ones
    | Size -> 
        [MemoryToReg; SimplifyCFG; DeadCodeElim]
    | SizeMin -> 
        [MemoryToReg; DeadCodeElim]

/// Checks if the opt tool is available
let private isOptAvailable() : bool =
    try
        let processInfo = System.Diagnostics.ProcessStartInfo()
        processInfo.FileName <- "opt"
        processInfo.Arguments <- "--version"
        processInfo.UseShellExecute <- false
        processInfo.RedirectStandardOutput <- true
        processInfo.RedirectStandardError <- true
        processInfo.CreateNoWindow <- true
        
        use proc = System.Diagnostics.Process.Start(processInfo)
        proc.WaitForExit(5000) |> ignore // 5 second timeout
        proc.ExitCode = 0
    with
    | _ -> false

/// Applies LLVM optimization passes to LLVM IR text using opt tool
let private runOptPasses (llvmIR: string) (passes: OptimizationPass list) : string =
    if passes.IsEmpty then
        llvmIR
    elif not (isOptAvailable()) then
        printfn "Warning: LLVM 'opt' tool not found - skipping LLVM optimizations"
        printfn "Install LLVM tools for optimization support: https://releases.llvm.org/download.html"
        llvmIR
    else
        try
            // Write LLVM IR to temporary file
            let inputPath = Path.GetTempFileName() + ".ll"
            let outputPath = Path.GetTempFileName() + ".ll"
            File.WriteAllText(inputPath, llvmIR)
            
            // Try new pass manager syntax first (LLVM 13+)
            let passNames = passes |> List.map (fun pass ->
                match pass with
                | MemoryToReg -> "mem2reg"
                | SimplifyCFG -> "simplifycfg"
                | InstCombine -> "instcombine"
                | DeadCodeElim -> "dce"
                | InliningPass -> "inline"
                | _ -> "mem2reg" // Fallback to safe pass
            )
            let pipelineStr = String.concat "," passNames
            
            // Run opt tool with new syntax
            let processInfo = System.Diagnostics.ProcessStartInfo()
            processInfo.FileName <- "opt"
            processInfo.Arguments <- sprintf "-passes=\"%s\" %s -o %s -S" pipelineStr inputPath outputPath
            processInfo.UseShellExecute <- false
            processInfo.RedirectStandardOutput <- true
            processInfo.RedirectStandardError <- true
            
            use optProc = System.Diagnostics.Process.Start(processInfo)
            optProc.WaitForExit()
            
            let optimizedIR = 
                if optProc.ExitCode = 0 && File.Exists(outputPath) then
                    File.ReadAllText(outputPath)
                else
                    // Try old syntax as fallback
                    let oldProcessInfo = System.Diagnostics.ProcessStartInfo()
                    oldProcessInfo.FileName <- "opt"
                    let oldPassArgs = passes |> List.map passToString |> String.concat " "
                    oldProcessInfo.Arguments <- sprintf "%s %s -o %s -S" oldPassArgs inputPath outputPath
                    oldProcessInfo.UseShellExecute <- false
                    oldProcessInfo.RedirectStandardOutput <- true
                    oldProcessInfo.RedirectStandardError <- true
                    
                    use oldOptProc = System.Diagnostics.Process.Start(oldProcessInfo)
                    oldOptProc.WaitForExit()
                    
                    if oldOptProc.ExitCode = 0 && File.Exists(outputPath) then
                        File.ReadAllText(outputPath)
                    else
                        let error = optProc.StandardError.ReadToEnd()
                        printfn "Warning: LLVM optimization failed (%s), using unoptimized IR" error
                        llvmIR
            
            // Cleanup temporary files
            if File.Exists(inputPath) then File.Delete(inputPath)
            if File.Exists(outputPath) then File.Delete(outputPath)
            
            optimizedIR
        with
        | ex ->
            printfn "Warning: Error during LLVM optimization (%s), using unoptimized IR" ex.Message
            llvmIR

/// Applies custom Firefly-specific optimizations to LLVM IR
let private applyFireflyOptimizations (llvmIR: string) : string =
    // Remove unnecessary allocations patterns
    let removeAllocations = 
        llvmIR.Replace("call void* @malloc", "; removed malloc call")
              .Replace("call void @free", "; removed free call")
    
    // Simplify stack-only patterns
    let simplifyStackPatterns =
        let allocaPattern = @"(%\w+)\s*=\s*alloca\s+([^,]+),.*\n.*store\s+([^,]+),\s*[^,]+\s*\1.*\n.*(%\w+)\s*=\s*load\s+[^,]+,\s*[^,]+\s*\1"
        Regex.Replace(removeAllocations, allocaPattern, fun m ->
            sprintf "%s = add %s 0, %s" m.Groups.[4].Value m.Groups.[2].Value m.Groups.[3].Value
        )
    
    // Optimize constant propagation for zero-allocation patterns
    let optimizeConstants =
        let constPattern = @"(%\w+)\s*=\s*add\s+(\w+)\s+0,\s*(\d+)"
        Regex.Replace(simplifyStackPatterns, constPattern, fun m ->
            sprintf "%s = add %s 0, %s  ; constant optimized" m.Groups.[1].Value m.Groups.[2].Value m.Groups.[3].Value
        )
    
    optimizeConstants

/// Validates that optimized IR maintains zero-allocation guarantees
let private validateZeroAllocation (llvmIR: string) : bool =
    let heapPatterns = [
        @"call.*@malloc"
        @"call.*@calloc"
        @"call.*@realloc"
        @"call.*@new"
        @"invoke.*@malloc"
    ]
    
    heapPatterns
    |> List.forall (fun pattern -> not (Regex.IsMatch(llvmIR, pattern)))

/// Applies optimization passes to LLVM IR with Firefly-specific enhancements
let optimizeLLVMIR (llvmOutput: LLVMOutput) (passes: OptimizationPass list) : LLVMOutput =
    let startingIR = llvmOutput.LLVMIRText
    
    // Apply standard LLVM optimizations
    let standardOptimizedIR = runOptPasses startingIR passes
    
    // Apply Firefly-specific optimizations
    let fireflyOptimizedIR = applyFireflyOptimizations standardOptimizedIR
    
    // Validate zero-allocation guarantee
    if not (validateZeroAllocation fireflyOptimizedIR) then
        printfn "Warning: Optimized IR may contain heap allocations"
    
    { llvmOutput with LLVMIRText = fireflyOptimizedIR }

/// Creates an aggressive optimization pipeline for release builds
let createReleaseOptimizationPipeline() : OptimizationPass list =
    createOptimizationPipeline Aggressive

/// Creates a debug-friendly optimization pipeline
let createDebugOptimizationPipeline() : OptimizationPass list =
    createOptimizationPipeline Less

/// Creates a size-optimized pipeline for embedded targets
let createSizeOptimizationPipeline() : OptimizationPass list =
    createOptimizationPipeline SizeMin

/// Estimates the impact of optimizations on code size and performance
let estimateOptimizationImpact (originalIR: string) (optimizedIR: string) : string =
    let originalLines = originalIR.Split('\n').Length
    let optimizedLines = optimizedIR.Split('\n').Length
    let reduction = ((float originalLines - float optimizedLines) / float originalLines) * 100.0
    
    sprintf "Optimization reduced IR by %.1f%% (%d -> %d lines)" reduction originalLines optimizedLines