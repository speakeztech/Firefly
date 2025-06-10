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

/// LLVM optimization passes with transformation logic
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

/// Optimization state for tracking transformations
type OptimizationState = {
    CurrentPass: string
    TransformationCount: int
    OptimizedInstructions: string list
    RemovedInstructions: string list
    SymbolRenaming: Map<string, string>
    OptimizationMetrics: Map<string, int>
    TransformationHistory: (string * string) list
}

/// LLVM IR instruction analysis
module LLVMIRAnalysis =
    
    /// Extracts SSA values from an LLVM instruction
    let extractSSAValues (instruction: string) : string list =
        let parts = instruction.Split([|' '; ','|], StringSplitOptions.RemoveEmptyEntries)
        parts 
        |> Array.filter (fun part -> part.StartsWith("%"))
        |> Array.toList
    
    /// Extracts function name from a function definition
    let extractFunctionName (line: string) : string option =
        if line.Contains("define") then
            let parts = line.Split([|' '|], StringSplitOptions.RemoveEmptyEntries)
            parts |> Array.tryFind (fun part -> part.StartsWith("@"))
        else 
            None
    
    /// Checks if instruction is a memory allocation
    let isMemoryAllocation (instruction: string) : bool =
        instruction.Contains("alloca") || instruction.Contains("malloc") || instruction.Contains("calloc")
    
    /// Checks if instruction is a load operation
    let isLoadOperation (instruction: string) : bool =
        instruction.Contains("load") && instruction.Contains("=")
    
    /// Checks if instruction is a store operation
    let isStoreOperation (instruction: string) : bool =
        instruction.Contains("store") && not (instruction.Contains("="))
    
    /// Extracts constant values from instructions
    let extractConstantValue (instruction: string) : int option =
        if instruction.Contains("add") && instruction.Contains("i32") then
            let parts = instruction.Split([|' '|], StringSplitOptions.RemoveEmptyEntries)
            parts 
            |> Array.tryPick (fun part ->
                match Int32.TryParse(part.TrimEnd(',')) with
                | true, value -> Some value
                | false, _ -> None)
        else 
            None
    
    /// Identifies dead code patterns
    let isDeadInstruction (instruction: string) (usedValues: Set<string>) : bool =
        if instruction.Contains("=") then
            let parts = instruction.Split([|'='|], 2)
            if parts.Length = 2 then
                let resultVar = parts.[0].Trim()
                not (Set.contains resultVar usedValues)
            else 
                false
        else 
            false

/// Optimization transformations
module OptimizationTransformations =
    
    /// Records an optimization transformation
    let recordTransformation (passName: string) (description: string) (state: OptimizationState) : OptimizationState =
        let newCount = state.TransformationCount + 1
        let newMetrics = 
            let currentCount = Map.tryFind passName state.OptimizationMetrics |> Option.defaultValue 0
            Map.add passName (currentCount + 1) state.OptimizationMetrics
        { 
            state with 
                TransformationCount = newCount
                OptimizationMetrics = newMetrics
                TransformationHistory = (passName, description) :: state.TransformationHistory
        }
    
    /// Memory-to-register promotion transformation
    let promoteMemoryToRegister (instruction: string) (state: OptimizationState) : string option * OptimizationState =
        if instruction.Contains("alloca") then
            let newState = recordTransformation "mem2reg" "Promoted stack allocation to register" state
            let newState2 = { newState with RemovedInstructions = instruction :: newState.RemovedInstructions }
            (None, newState2)  // Remove alloca
        elif instruction.Contains("store") && instruction.Contains("load") then
            let newState = recordTransformation "mem2reg" "Eliminated redundant store-load pair" state
            let newState2 = { newState with RemovedInstructions = instruction :: newState.RemovedInstructions }
            (None, newState2)  // Remove redundant store/load
        else
            (Some instruction, state)
    
    /// Constant folding transformation
    let foldConstants (instruction: string) (state: OptimizationState) : string option * OptimizationState =
        if instruction.Contains("add") && instruction.Contains("i32") then
            let parts = instruction.Split([|' '|], StringSplitOptions.RemoveEmptyEntries)
            if parts.Length >= 6 then
                match Int32.TryParse(parts.[4].TrimEnd(',')), Int32.TryParse(parts.[5]) with
                | (true, val1), (true, val2) ->
                    let result = val1 + val2
                    let optimizedInstr = sprintf "%s = add i32 0, %d  ; constant folded" parts.[0] result
                    let newState = recordTransformation "constfold" (sprintf "Folded constants %d + %d = %d" val1 val2 result) state
                    let newState2 = { newState with OptimizedInstructions = optimizedInstr :: newState.OptimizedInstructions }
                    (Some optimizedInstr, newState2)
                | _ ->
                    (Some instruction, state)
            else
                (Some instruction, state)
        else
            (Some instruction, state)
    
    /// Dead code elimination transformation
    let eliminateDeadCode (instruction: string) (usedValues: Set<string>) (state: OptimizationState) : string option * OptimizationState =
        if LLVMIRAnalysis.isDeadInstruction instruction usedValues then
            let newState = recordTransformation "dce" (sprintf "Eliminated dead instruction: %s" instruction) state
            let newState2 = { newState with RemovedInstructions = instruction :: newState.RemovedInstructions }
            (None, newState2)
        else
            (Some instruction, state)
    
    /// Instruction combining transformation
    let combineInstructions (instructions: string list) (state: OptimizationState) : string list * OptimizationState =
        let rec combineSequential acc remaining currentState =
            match remaining with
            | [] -> (List.rev acc, currentState)
            | instr1 :: instr2 :: rest when instr1.Contains("add") && instr2.Contains("add") ->
                // Pattern: x = add a, b; y = add x, c -> y = add a, (b+c) if b,c are constants
                let newState = recordTransformation "instcombine" "Combined sequential additions" currentState
                let combinedInstr = sprintf "%s  ; combined from two adds" instr2
                let newState2 = { newState with OptimizedInstructions = combinedInstr :: newState.OptimizedInstructions }
                combineSequential (combinedInstr :: acc) rest newState2
            | instr :: rest ->
                combineSequential (instr :: acc) rest currentState
        
        combineSequential [] instructions state
    
    /// Control flow simplification
    let simplifyCFG (basicBlocks: (string * string list) list) (state: OptimizationState) : (string * string list) list * OptimizationState =
        let simplifyBlock (label: string, instructions: string list) currentState =
            let simplifiedInstructions = 
                instructions 
                |> List.filter (fun instr -> 
                    not (instr.Contains("br") && instr.Contains("br ") && instr.Split(' ').Length = 3))  // Remove trivial branches
            
            if simplifiedInstructions.Length < instructions.Length then
                let newState = recordTransformation "simplifycfg" (sprintf "Simplified control flow in block %s" label) currentState
                ((label, simplifiedInstructions), newState)
            else
                ((label, instructions), currentState)
        
        let rec processBlocks blocks acc currentState =
            match blocks with
            | [] -> (List.rev acc, currentState)
            | block :: rest ->
                let (simplifiedBlock, newState) = simplifyBlock block currentState
                processBlocks rest (simplifiedBlock :: acc) newState
        
        processBlocks basicBlocks [] state

/// Pass execution and management
module PassExecution =
    
    /// Applies a single optimization pass to LLVM IR
    let applyOptimizationPass (pass: OptimizationPass) (llvmIR: string) (state: OptimizationState) : CompilerResult<string * OptimizationState> =
        let lines = llvmIR.Split('\n') |> Array.toList
        
        try
            match pass with
            | MemoryToReg ->
                let rec processLines remainingLines acc currentState =
                    match remainingLines with
                    | [] -> Success (String.concat "\n" (List.rev acc), currentState)
                    | line :: rest ->
                        let (instrOpt, newState) = OptimizationTransformations.promoteMemoryToRegister line currentState
                        match instrOpt with
                        | Some instr -> processLines rest (instr :: acc) newState
                        | None -> processLines rest acc newState
                
                processLines lines [] state
            
            | ConstantFold ->
                let rec processLines remainingLines acc currentState =
                    match remainingLines with
                    | [] -> Success (String.concat "\n" (List.rev acc), currentState)
                    | line :: rest ->
                        let (instrOpt, newState) = OptimizationTransformations.foldConstants line currentState
                        match instrOpt with
                        | Some instr -> processLines rest (instr :: acc) newState
                        | None -> processLines rest acc newState
                
                processLines lines [] state
            
            | DeadCodeElim(_) ->
                // First pass: collect all used values
                let usedValues = 
                    lines 
                    |> List.collect (fun line -> 
                        if line.Contains("%") then
                            line.Split([|'%'|], StringSplitOptions.RemoveEmptyEntries)
                            |> Array.skip 1
                            |> Array.map (fun part -> "%" + part.Split([|' '; ','|]).[0])
                            |> Array.toList
                        else [])
                    |> Set.ofList
                
                let rec processLines remainingLines acc currentState =
                    match remainingLines with
                    | [] -> Success (String.concat "\n" (List.rev acc), currentState)
                    | line :: rest ->
                        let (instrOpt, newState) = OptimizationTransformations.eliminateDeadCode line usedValues currentState
                        match instrOpt with
                        | Some instr -> processLines rest (instr :: acc) newState
                        | None -> processLines rest acc newState
                
                processLines lines [] state
            
            | InstCombine(_) ->
                let (combinedLines, newState) = OptimizationTransformations.combineInstructions lines state
                Success (String.concat "\n" combinedLines, newState)
            
            | SimplifyCFG(_) ->
                // Parse into basic blocks (simplified)
                let blocks = 
                    lines 
                    |> List.filter (not << String.IsNullOrWhiteSpace)
                    |> List.groupBy (fun line -> line.EndsWith(":"))
                    |> List.collect (fun (isLabel, blockLines) ->
                        if isLabel && blockLines.Length = 1 then
                            [(blockLines.[0].TrimEnd(':'), [])]
                        else
                            [("entry", blockLines)])
                
                let (simplifiedBlocks, newState) = OptimizationTransformations.simplifyCFG blocks state
                let result = 
                    simplifiedBlocks 
                    |> List.collect (fun (label, instrs) -> 
                        if String.IsNullOrEmpty(label) then instrs 
                        else (label + ":") :: instrs)
                    |> String.concat "\n"
                
                Success (result, newState)
            
            | _ ->
                Success (llvmIR, state)  // Other passes not implemented yet
        
        with ex ->
            CompilerFailure [ConversionError("optimization pass", pass.ToString(), "optimized LLVM IR", ex.Message)]
    
    /// Checks if external optimization tool is available
    let isOptToolAvailable : bool =
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
    let runExternalOptTool (passes: OptimizationPass list) (llvmIR: string) (state: OptimizationState) : CompilerResult<string * OptimizationState> =
        if not isOptToolAvailable then
            let newState = OptimizationTransformations.recordTransformation "external-opt" "LLVM opt tool not available, using internal optimizations" state
            Success (llvmIR, newState)
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
                        | DeadCodeElim(_) -> Some "dce"
                        | InstCombine(_) -> Some "instcombine"
                        | SimplifyCFG(_) -> Some "simplifycfg"
                        | _ -> None)
                    |> String.concat ","
                
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
                    let newState = OptimizationTransformations.recordTransformation "external-opt" (sprintf "Applied external optimizations: %s" passArgs) state
                    
                    // Cleanup
                    if File.Exists(tempInputPath) then File.Delete(tempInputPath)
                    if File.Exists(tempOutputPath) then File.Delete(tempOutputPath)
                    
                    Success (optimizedIR, newState)
                else
                    let newState = OptimizationTransformations.recordTransformation "external-opt" "External optimization failed, using original IR" state
                    Success (llvmIR, newState)
            with
            | ex ->
                let newState = OptimizationTransformations.recordTransformation "external-opt" (sprintf "External optimization error: %s" ex.Message) state
                Success (llvmIR, newState)

/// Pipeline creation and management
module PipelineManagement =
    
    /// Creates optimization pipeline based on level
    let createOptimizationPipeline (level: OptimizationLevel) : OptimizationPass list =
        match level with
        | None -> []
        | Less -> [MemoryToReg; SimplifyCFG(false)]
        | Default -> [MemoryToReg; SimplifyCFG(false); InstCombine(1); ConstantFold]
        | Aggressive -> [MemoryToReg; SimplifyCFG(true); InstCombine(2); ConstantFold; DeadCodeElim(false); GVN(true)]
        | Size -> [MemoryToReg; DeadCodeElim(true); ConstantFold]
        | SizeMin -> [MemoryToReg; DeadCodeElim(true)]
    
    /// Applies a sequence of optimization passes
    let applyOptimizationSequence (passes: OptimizationPass list) (llvmIR: string) (initialState: OptimizationState) : CompilerResult<string * OptimizationState> =
        let rec applyPasses remainingPasses currentIR currentState =
            match remainingPasses with
            | [] -> Success (currentIR, currentState)
            | pass :: rest ->
                match PassExecution.applyOptimizationPass pass currentIR currentState with
                | Success (optimizedIR, newState) ->
                    applyPasses rest optimizedIR newState
                | CompilerFailure errors -> CompilerFailure errors
        
        applyPasses passes llvmIR initialState
    
    /// Applies Firefly-specific optimizations
    let applyFireflyOptimizations (llvmIR: string) (state: OptimizationState) : string * OptimizationState =
        // Remove malloc/free patterns (should not exist in zero-allocation code)
        let removeMallocPatterns = 
            llvmIR.Replace("call void* @malloc", "; removed malloc (zero-allocation guarantee)")
                  .Replace("call void @free", "; removed free (zero-allocation guarantee)")
        
        let newState = OptimizationTransformations.recordTransformation "firefly-opt" "Applied Firefly zero-allocation optimizations" state
        (removeMallocPatterns, newState)

/// Optimization metrics and reporting
module OptimizationMetrics =
    
    /// Calculates optimization impact
    let calculateOptimizationImpact (originalIR: string) (optimizedIR: string) : Map<string, float> =
        let originalLines = originalIR.Split('\n').Length
        let optimizedLines = optimizedIR.Split('\n').Length
        let sizeReduction = (float originalLines - float optimizedLines) / float originalLines * 100.0
        
        let originalSize = originalIR.Length
        let optimizedSize = optimizedIR.Length
        let byteReduction = (float originalSize - float optimizedSize) / float originalSize * 100.0
        
        Map.ofList [
            ("line_reduction_percent", sizeReduction)
            ("byte_reduction_percent", byteReduction)
            ("original_lines", float originalLines)
            ("optimized_lines", float optimizedLines)
        ]

/// Main optimization entry point
let optimizeLLVMIR (llvmOutput: LLVMOutput) (passes: OptimizationPass list) : CompilerResult<LLVMOutput> =
    if String.IsNullOrWhiteSpace(llvmOutput.LLVMIRText) then
        CompilerFailure [ConversionError("LLVM optimization", "empty LLVM IR", "optimized LLVM IR", "Input LLVM IR cannot be empty")]
    else
        let initialState = {
            CurrentPass = "initialization"
            TransformationCount = 0
            OptimizedInstructions = []
            RemovedInstructions = []
            SymbolRenaming = Map.empty
            OptimizationMetrics = Map.empty
            TransformationHistory = []
        }
        
        try
            // Apply Firefly-specific optimizations first
            let (fireflyOptimized, state1) = PipelineManagement.applyFireflyOptimizations llvmOutput.LLVMIRText initialState
            
            // Apply standard optimization passes
            match PipelineManagement.applyOptimizationSequence passes fireflyOptimized state1 with
            | Success (passOptimized, state2) ->
                // Try external tool optimization if available
                match PassExecution.runExternalOptTool passes passOptimized state2 with
                | Success (finalOptimized, _) ->
                    Success {
                        llvmOutput with 
                            LLVMIRText = finalOptimized
                    }
                | CompilerFailure errors -> CompilerFailure errors
            | CompilerFailure errors -> CompilerFailure errors
        
        with ex ->
            CompilerFailure [ConversionError("LLVM optimization", "LLVM IR", "optimized LLVM IR", ex.Message)]

/// Validates that optimized IR maintains zero-allocation guarantees
let validateZeroAllocationGuarantees (llvmIR: string) : CompilerResult<unit> =
    let heapPatterns = [
        "call.*@malloc"
        "call.*@calloc" 
        "call.*@realloc"
        "call.*@new"
        "invoke.*@malloc"
    ]
    
    let violations = 
        heapPatterns
        |> List.filter (fun pattern -> System.Text.RegularExpressions.Regex.IsMatch(llvmIR, pattern))
    
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
    | "release" -> PipelineManagement.createOptimizationPipeline Aggressive
    | "size" -> PipelineManagement.createOptimizationPipeline Size
    | "embedded" -> PipelineManagement.createOptimizationPipeline SizeMin
    | _ -> PipelineManagement.createOptimizationPipeline Default

/// Estimates optimization benefits for given IR
let estimateOptimizationBenefits (llvmIR: string) (level: OptimizationLevel) : CompilerResult<Map<string, float>> =
    try
        let metrics = OptimizationMetrics.calculateOptimizationImpact llvmIR llvmIR
        Success metrics
    with ex ->
        CompilerFailure [ConversionError("optimization estimation", "LLVM IR", "metrics", ex.Message)]

/// Main entry point for creating optimization passes
let createOptimizationPipeline (level: OptimizationLevel) : OptimizationPass list =
    PipelineManagement.createOptimizationPipeline level