module Alex.Lite.LitePipeline

open System
open System.IO
open Core.PSG.Types
open Alex.Lite.SimpleLLVMEmitter

/// Configuration for the lite pipeline
type LitePipelineConfig = {
    OutputDir: string
    GenerateIntermediate: bool
    RunMLIROpt: bool
    TargetTriple: string option
}

/// Result of lite pipeline execution
type LitePipelineResult = {
    Success: bool
    MLIROutput: string option
    LLVMOutput: string option
    Errors: string list
}

/// Process PSG through simplified LLVM-only pipeline
let processWithLitePipeline (psg: ProgramSemanticGraph) (config: LitePipelineConfig) =
    try
        printfn "[LitePipeline] Starting simplified LLVM dialect generation..."
        
        // Ensure output directory exists
        if not (Directory.Exists(config.OutputDir)) then
            Directory.CreateDirectory(config.OutputDir) |> ignore
        
        // Generate LLVM dialect MLIR
        let mlirPath = Path.Combine(config.OutputDir, "output.mlir")
        emitLLVMDialect psg mlirPath
        
        // Optionally run mlir-opt to lower to LLVM IR
        let llvmPath = 
            if config.RunMLIROpt then
                let llPath = Path.Combine(config.OutputDir, "output.ll")
                
                // Build mlir-opt command
                // Note: This assumes mlir-opt is in PATH
                let mlirOptCmd = 
                    sprintf "mlir-opt --convert-func-to-llvm --reconcile-unrealized-casts %s | mlir-translate --mlir-to-llvmir -o %s" 
                        mlirPath llPath
                
                printfn "[LitePipeline] Running: %s" mlirOptCmd
                
                // Execute command (simplified - in production you'd want proper process handling)
                let psi = System.Diagnostics.ProcessStartInfo()
                psi.FileName <- "cmd.exe"
                psi.Arguments <- sprintf "/c %s" mlirOptCmd
                psi.UseShellExecute <- false
                psi.RedirectStandardOutput <- true
                psi.RedirectStandardError <- true
                
                use proc = System.Diagnostics.Process.Start(psi)
                proc.WaitForExit()
                
                if proc.ExitCode = 0 then
                    printfn "[LitePipeline] Successfully generated LLVM IR"
                    Some llPath
                else
                    let error = proc.StandardError.ReadToEnd()
                    printfn "[LitePipeline] mlir-opt failed: %s" error
                    None
            else
                None
        
        // Generate intermediate files if requested
        if config.GenerateIntermediate then
            let intermediatePath = Path.Combine(config.OutputDir, "intermediate.mlir")
            generateMinimalExample intermediatePath
        
        {
            Success = true
            MLIROutput = Some mlirPath
            LLVMOutput = llvmPath
            Errors = []
        }
        
    with ex ->
        printfn "[LitePipeline] Error: %s" ex.Message
        {
            Success = false
            MLIROutput = None
            LLVMOutput = None
            Errors = [ex.Message]
        }

/// Create a simple test PSG for validation
let createTestPSG() =
    let nodes = 
        [
            // Main function node
            { PSGNode.Empty with
                Id = 1
                NodeId = "main_func"
                SyntaxKind = "Binding"
                Symbol = Some {
                    SymbolInformation.Empty with
                        Name = "main"
                        FullName = "HelloWorld.main"
                        IsFunction = true
                }
                IsReachable = true
                Children = Parent [NodeId 2; NodeId 3]
            }
            
            // Hello function call
            { PSGNode.Empty with
                Id = 2
                NodeId = "hello_call"
                SyntaxKind = "FunctionCall"
                Symbol = Some {
                    SymbolInformation.Empty with
                        Name = "hello"
                        FullName = "HelloWorld.hello"
                        IsFunction = true
                }
                IsReachable = true
            }
            
            // Return statement
            { PSGNode.Empty with
                Id = 3
                NodeId = "return_zero"
                SyntaxKind = "Const"
                ConstantValue = Some "0"
                TypeName = Some "int"
                IsReachable = true
            }
            
            // Hello function node
            { PSGNode.Empty with
                Id = 4
                NodeId = "hello_func"
                SyntaxKind = "Binding"
                Symbol = Some {
                    SymbolInformation.Empty with
                        Name = "hello"
                        FullName = "HelloWorld.hello"
                        IsFunction = true
                }
                IsReachable = true
                Children = Parent [NodeId 5]
            }
            
            // WriteLine call
            { PSGNode.Empty with
                Id = 5
                NodeId = "writeline_call"
                SyntaxKind = "MethodCall"
                Symbol = Some {
                    SymbolInformation.Empty with
                        Name = "WriteLine"
                        FullName = "System.Console.WriteLine"
                        IsFunction = true
                }
                IsReachable = true
            }
        ]
        |> List.map (fun node -> (node.Id, node))
        |> Map.ofList
    
    {
        Nodes = nodes
        Edges = []
        EntryPoints = Set.singleton "main"
        Metadata = Map.empty
    }

/// Quick test function
let testLitePipeline() =
    printfn "[LitePipeline] Running test..."
    
    let testPSG = createTestPSG()
    let config = {
        OutputDir = "./test_output"
        GenerateIntermediate = true
        RunMLIROpt = false  // Set to true if you have MLIR tools installed
        TargetTriple = None
    }
    
    let result = processWithLitePipeline testPSG config
    
    if result.Success then
        printfn "[LitePipeline] Test successful!"
        match result.MLIROutput with
        | Some path -> 
            printfn "  MLIR output: %s" path
            let content = File.ReadAllText(path)
            printfn "  Content preview:"
            printfn "%s" (content.Substring(0, min 500 content.Length))
        | None -> ()
    else
        printfn "[LitePipeline] Test failed: %A" result.Errors